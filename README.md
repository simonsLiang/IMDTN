#IMDTN
code for NTIRE 2022 Efficient Super-Resolution Challenge

When calculating 'flops',' activations', 'conv', and 'max memory allocated', the height and width of the input dimension must be adjusted to a multiple of six first.

`input_dim = (3,256,256)`
`H,W = input_dim[-2],input_dim[-1]`
`input_dim = (input_dim[0],(H//6+1)*6,(W//6+1)*6)  #(3,258,258)`
