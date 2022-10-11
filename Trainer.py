import torch
from torch import nn
import tqdm


def Train(model, config):
    # epochs = 5
    # train loop
    model = model.to(config.device)

    epochloop = tqdm(range(config.epochs), position=0, desc='Training', leave=True)

    # early stop trigger
    es_trigger = 0
    val_loss_min = torch.inf

    for e in epochloop:

        #################
        # training mode #
        #################

        model.train()

        train_loss = 0
        train_acc = 0

        for id, (feature, target) in enumerate(trainloader):
            # add epoch meta info
            epochloop.set_postfix_str(f'Training batch {id}/{len(trainloader)}')

            # move to device
            feature, target = feature.to(config.device), target.to(config.device)

            # reset optimizer
            optim.zero_grad()

            # forward pass
            out = model(feature)

            # acc
            predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=config.device)
            equals = predicted == target
            acc = torch.mean(equals.type(torch.FloatTensor))
            train_acc += acc.item()
            # loss
            # print(out.squeeze().shape)
            # print(target.shape)
            loss = criterion(out.squeeze(), target.float())
            train_loss += loss.item()
            loss.backward()

            # clip grad
            # nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # update optimizer
            optim.step()

            # free some memory
            del feature, target, predicted

        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc / len(trainloader))

        ####################
        # validation model #
        ####################

        model.eval()

        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for id, (feature, target) in enumerate(valloader):
                # add epoch meta info
                epochloop.set_postfix_str(f'Validation batch {id}/{len(valloader)}')

                # move to device
                feature, target = feature.to(config.device), target.to(config.device)

                # forward pass
                out = model(feature)

                # acc
                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=config.device)
                equals = predicted == target
                acc = torch.mean(equals.type(torch.FloatTensor))
                val_acc += acc.item()

                # loss
                loss = criterion(out.squeeze(), target.float())
                val_loss += loss.item()

                # free some memory
                del feature, target, predicted

            history['val_loss'].append(val_loss / len(valloader))
            history['val_acc'].append(val_acc / len(valloader))

        # reset model mode
        model.train()

        # add epoch meta info
        epochloop.set_postfix_str(f'Val Loss: {val_loss / len(valloader):.3f} | Val Acc: {val_acc / len(valloader):.3f}')

        # print epoch
        if (e+1) % print_every == 0:
            epochloop.write(f'Epoch {e+1}/{epochs} | Train Loss: {train_loss / len(trainloader):.3f} Train Acc: {train_acc / len(trainloader):.3f} | Val Loss: {val_loss / len(valloader):.3f} Val Acc: {val_acc / len(valloader):.3f}')
            epochloop.update()

        # save model if validation loss decrease
        if val_loss / len(valloader) <= val_loss_min:
            torch.save(model.state_dict(), './sentiment_lstm.pt')
            val_loss_min = val_loss / len(valloader)
            es_trigger = 0
        else:
            epochloop.write(f'[WARNING] Validation loss did not improved ({val_loss_min:.3f} --> {val_loss / len(valloader):.3f})')
            es_trigger += 1

        # force early stop
        if es_trigger >= es_limit:
            epochloop.write(f'Early stopped at Epoch-{e+1}')
            # update epochs history
            history['epochs'] = e+1
            break