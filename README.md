# DUE

Equations, particularly differential equations, are fundamental for understanding natural phenomena and predicting complex dynamics across various scientific and engineering disciplines. However, the governing equations for many complex systems remain unknown due to intricate underlying mechanisms. Recent advancements in machine learning and data science offer a new paradigm for modeling unknown equations from measurement or simulation data. This paradigm shift, known as data-driven discovery or modeling, stands at the forefront of artificial intelligence for science (AI4Science), with significant progress made in recent years. 

Deep Unknown Equations (DUE) is an open-source software package designed to facilitate the data-driven modeling of unknown equations using modern deep learning techniques. This versatile framework is capable of learning unknown ordinary differential equations (ODEs), partial differential equations (PDEs), differential-algebraic equations (DAEs), integro-differential equations (IDEs), stochastic differential equations (SDEs), reduced or partially observed systems, and non-autonomous differential equations.

DUE serves as an educational tool for classroom instruction, enabling students and newcomers to gain hands-on experience with differential equations, data-driven modeling, and contemporary deep learning approaches such as fully connected neural networks (FNN), residual neural networks (ResNet), generalized ResNet (gResNet), operator semigroup networks (OSG-Net), and Transformers from large language models (LLMs). Additionally, DUE is a versatile and accessible toolkit for researchers across various scientific and engineering fields. It is applicable not only for learning unknown equations from data but also for surrogate modeling of known, yet complex, equations that are costly to solve using traditional numerical methods. We provide detailed descriptions of DUE and demonstrate its capabilities through diverse examples, which serve as templates that can be easily adapted for other applications.

## Installation

DUE, along with all its dependencies, can be easily installed through running the 'setup.py' file:

``` sh
$ python setup.py install
```

## Demonstration examples

We provide a wide range of examples to show the usage and customizability of DUE.  inside 'examples' for detailed 