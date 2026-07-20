import sys
from tqdm.auto import tqdm
import torch

def data_list_to_predict_gen(data):
    for frame in data:
        yield  torch.unsqueeze(frame, 0)

EPSILON = 1e-7
def predict(pipliner, generator, device=None, eps=EPSILON, desc_tqdm="\t\tPredict"):
    model = pipliner.model
    model.to(device)
    model.eval()
    result = []

    with torch.no_grad():
        tqdm_test_loop = tqdm(generator,
                              #leave=False,
                              file=sys.stdout,
                              desc=desc_tqdm,
                              colour="GREEN",
                              disable=False,
                              #leave=False,
                              #position=1
                              )

        for inputs in tqdm_test_loop:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # ADD LAST ACTIVATION
            outputs = pipliner.last_activation_fun(outputs, eps)
            result.append(outputs[0]) # no batch
    return result