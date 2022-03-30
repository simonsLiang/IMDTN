# IMDTN

## Code for NTIRE 2022 Efficient Super-Resolution Challenge

When calculating 'flops',' activations', 'conv', and 'max memory allocated', the height and width of the input should be a multiple of six.

If the input size does not meet the requirements, it can be adjusted in the following ways:

`input_dim = (3,256,256)`

`H,W = input_dim[-2],input_dim[-1]`

`input_dim = (input_dim[0],(H//6+1)*6,(W//6+1)*6)  #(3,258,258)`

`from torchvision import transforms`

`aug = transforms.Resize((input_dim[-2],input_dim[-1])) #Used to change the input size` 

And `main_challenge_sr.py` is provided to calculate these metrics
