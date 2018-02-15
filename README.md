# ConsciousQuestion

snip_lables
  1,000 of state 2
  1,000 of state 1

data is the associated brain state at that point in time.

  feed in 100 samples, predict the classification at the end.
  this is anaagous to 



  if (curEpoch % 10 == 0 or curEpoch == 1):
                    #Calculate the validation loss
                    val_loss = sess.run(cost,feed_dict={X:val_X,y:val_Y})
                    print("Validation loss: " , str(val_loss))
                    val_loss_file.write(str(float(val_loss))+str("\n"))
                    val_loss_file.flush()

                    print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss))
                    train_loss_file.flush()
                    