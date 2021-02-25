# Quantization

~~Despite best efforts, due to the limitations of tfmot, it is NOT possible to create a quantization aware model at this point.
This is because the transformer abstracttions do not support layer reuse, and does not support nested models,
nor nestsed layers. Basically, nothing is supported with our architecture and there is no feasible pathway for using tfmot.~~

Quantization works now, but there's a pretty significant architectural constraint:
namely, `qpwcnet.core.non_layers` module has to be used in place of `qpwcnet.core.layers`.
This essentially replaces all custom compositional layers into equivalent functors, thereby removing the nesting in the hierarchy.
