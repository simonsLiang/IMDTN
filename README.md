#IMDTN
code for NTIRE 2022 Efficient Super-Resolution Challenge

When calculating 'flops',' activations', 'conv', and 'max memory allocated', the height and width of the input dimension must be adjusted to a multiple of six first.

`input_dim = (3,256,256)`

`H,W = input_dim[-2],input_dim[-1]`

`window_size = 6`

`if H % window_size != 0:`

  `H = (H//window_size+1)*window_size`
  
`if W % window_size != 0:`

  `W = (W//window_size+1)*window_size`
  
`input_dim = (input_dim[0],H,W)  #(3,258,258)`
