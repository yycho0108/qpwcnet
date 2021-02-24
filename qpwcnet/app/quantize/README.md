# Quantization

Despite best efforts, due to the limitations of tfmot, it is NOT possible to create a quantization aware model at this point.

This is because the transformer abstracttions do not support layer reuse, and does not support nested models,

nor nestsed layers. Basically, nothing is supported with our architecture and there is no feasible pathway for using tfmot.
