# Don't forget to support cases when target_text == ''

import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if target_text == '':
        if predicted_text == '':
            return 0
        else:
            return 1
    return editdistance.eval(target_text, predicted_text) / len(target_text)    


def calc_wer(target_text, predicted_text) -> float:
    if target_text == '':
        if predicted_text == '':
            return 0
        else:
            return 1
    target = target_text.split()
    predict = predicted_text.split()
    return editdistance.eval(target, predict) / len(target) 
 
    