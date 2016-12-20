# HDR-Fast-Local-Laplacian
HDR recovering and tonemapping using fast local laplacian approach

HDR recovering
paper
http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf
implementation
http://cybertron.cg.tu-berlin.de/eitz/hdr/#downloads

Tone mapping and manipulation
Local laplacian filter: 
paper and implementation
https://people.csail.mit.edu/sparis/publi/2011/siggraph/

Fast local laplacian filter:
paper and implementation
http://www.di.ens.fr/~aubry/llf.html

This is a matlab code that can build HDR from brackets, and do fast local laplacian tone mapping with
some adjustable varibales like alpha(decide details), beta(decide contrast), number of references. 
No Gaussian pyramids pregenerating for memory consideration.

![result](https://cloud.githubusercontent.com/assets/16308037/21352608/381206d0-c6fd-11e6-979b-1ba80b26a5e9.png)
