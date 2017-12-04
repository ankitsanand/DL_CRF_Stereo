import numpy as np

error_px = 3
error_percent = 0.05

def compute_error(epoch, batch_num, log_file, loss, prediction, gt):
  diff = np.abs(prediction - gt)
  # wrong_elems = (gt > 0) & (diff > error_px) & ((diff / np.abs(gt)) > error_percent)
  wrong_elems = (gt > 0) & (diff > error_px)

  error = 100 * wrong_elems.sum() / np.sum(gt[gt > 0].shape).astype('float64')
  log = "epoch "+str(epoch)+" batch "+str(batch_num)+" error "+str(error)+" loss "+str(loss)
  print(log)
  log_file.write(log+"\n")
  log_file.flush()
  return error
