import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.metric.utils import calc_cer, calc_wer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = {}
    wers = {}
    cers = {}

    with torch.no_grad():
        for part in ["test-clean", "test-other"]:
            results[part] = []
            wer_argmax = 0.0
            cer_argmax = 0.0
            wer_beam_search = 0.0
            cer_beam_search = 0.0
            count = 0
            for batch_num, batch in enumerate(tqdm(dataloaders[part])):
                batch = Trainer.move_batch_to_device(batch, device)
                output = model(**batch)
                if type(output) is dict:
                    batch.update(output)
                else:
                    batch["logits"] = output
                batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
                batch["log_probs_length"] = model.transform_input_lengths(
                    batch["spectrogram_length"]
                )
                batch["probs"] = batch["log_probs"].exp().cpu()
                batch["argmax"] = batch["probs"].argmax(-1)
                for i in range(len(batch["text"])):
                    argmax = batch["argmax"][i]
                    argmax = argmax[: int(batch["log_probs_length"][i])]
                    gt = batch["text"][i]
                    pred_argmax = text_encoder.ctc_decode(argmax.cpu().numpy())
                    pred_ctc_beam_search = text_encoder.ctc_beam_search(
                                              batch["probs"][i], batch["log_probs_length"][i], beam_size=5
                                              )[:5]
                    wer_argmax += calc_wer(gt, pred_argmax)
                    cer_argmax += calc_cer(gt, pred_argmax)
                    wer_beam_search += calc_wer(gt, pred_ctc_beam_search[0][0])
                    cer_beam_search += calc_cer(gt, pred_ctc_beam_search[0][0])
                    count += 1
                    results[part].append(
                        {
                            "ground_trurh": gt,
                            "pred_text_argmax": pred_argmax,
                            "pred_text_beam_search_first": pred_ctc_beam_search[0][0],
                        }
                    )
            wer_argmax /= count
            cer_argmax /= count
            wer_beam_search /= count
            cer_beam_search /= count
            wers[part] = [wer_argmax, wer_beam_search]
            cers[part] = [cer_argmax, cer_beam_search]

    for part in ['test-clean', 'test-other']:
        with Path(out_file[:-5]+'-'+part+out_file[-5:]).open("w") as f:
            json.dump(results[part], f, indent=2)
        logger.info("{}: wer (argmax) = {}, wer (ctc_beam_search) = {}, cer (argmax) = {}, cer (ctc_beam_search) = {},".format(
            part, wers[part][0], wers[part][1], cers[part][0], cers[part][1]))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    # args.add_argument(
    #     "-t",
    #     "--test-data-folder",
    #     default=None,
    #     type=str,
    #     help="Path to dataset",
    # )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    # if args.test_data_folder is not None:
    #     test_data_folder = Path(args.test_data_folder).absolute().resolve()
    #     assert test_data_folder.exists()
    #     config.config["data"] = {
    #         "test": {
    #             "batch_size": args.batch_size,
    #             "num_workers": args.jobs,
    #             "datasets": [
    #                 {
    #                     "type": "CustomDirAudioDataset",
    #                     "args": {
    #                         "audio_dir": str(test_data_folder / "audio"),
    #                         "transcription_dir": str(
    #                             test_data_folder / "transcriptions"
    #                         ),
    #                     },
    #                 }
    #             ],
    #         }
    #     }

    assert config.config.get("data", {}).get("test-clean", None) is not None
    assert config.config.get("data", {}).get("test-other", None) is not None
    config["data"]["test-clean"]["batch_size"] = args.batch_size
    config["data"]["test-other"]["batch_size"] = args.batch_size
    config["data"]["test-clean"]["n_jobs"] = args.jobs
    config["data"]["test-other"]["n_jobs"] = args.jobs

    main(config, args.output)
