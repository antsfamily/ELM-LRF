
# ELM-LRF
An Implementation of Local Receptive Fields Based Extreme Learning Machine

Strictly for academic use.

Please kindly cite the paper "Local Receptive Fields based Extreme Learning Machine".

Huang G, Bai Z, Kasun L, et al. Local Receptive Fields Based  Extreme Learning Machine[J]. Computational Intelligence Magazine dsa0987654321`IEEE,  2015, 10(2):18 - 29.

# An Outline of the paper

Refer to my [blog](http://blog.csdn.net/enjoyyl/article/details/45724367) to see more information.


Any bug report or suggestions are gladly welcome！Thanks！

Email：zhiliu.mind@gmail.com

# Usage

Please see the demos for detail information.

There are two main model for choosing, one is sequential(by setting `opts.model = sequential`):

![Sequential model](http://img.blog.csdn.net/20160312174915472 "Sequential model")

another one is parallel(by setting `opts.model = parallel`):
![Parallel model](http://img.blog.csdn.net/20160312175341467 "Parallel model")


# Experimental Result

Setting for Experiment:

- Dataset: NORB
- kernelsize: 4*4
- the number of convolutional feature maps: 3
- pooling size: 3*3


```matlab
With C = 0.001000
-----------------------------------------
Training error: 0.027284
Training Time:61.468750s

Testing error: 0.114856
Testing Time:14.625000s

With C = 0.010000
-----------------------------------------
Training error: 0.008272
Training Time:64.171875s

Testing error: 0.088724
Testing Time:13.921875s

With C = 0.100000
-----------------------------------------
Training error: 0.001564
Training Time:61.515625s

Testing error: 0.104033
Testing Time:13.281250s

With C = 0.200000
-----------------------------------------
Training error: 0.001070
Training Time:60.546875s

Testing error: 0.111111
Testing Time:12.625000s

With C = 0.300000
-----------------------------------------
Training error: 0.000947
Training Time:58.468750s

Testing error: 0.116831
Testing Time:12.765625s

With C = 0.400000
-----------------------------------------
Training error: 0.000741
Training Time:56.328125s

Testing error: 0.121152
Testing Time:12.281250s

With C = 0.500000
-----------------------------------------
Training error: 0.000700
Training Time:56.671875s

Testing error: 0.123498
Testing Time:12.765625s

With C = 0.600000
-----------------------------------------
Training error: 0.000658
Training Time:61.468750s

Testing error: 0.125021
Testing Time:13.328125s

With C = 0.700000
-----------------------------------------
Training error: 0.000617
Training Time:58.515625s

Testing error: 0.126584
Testing Time:12.828125s

With C = 0.800000
-----------------------------------------
Training error: 0.000576
Training Time:59.359375s

Testing error: 0.127984
Testing Time:12.796875s

With C = 0.900000
-----------------------------------------
Training error: 0.000535
Training Time:60.296875s

Testing error: 0.129136
Testing Time:12.156250s

With C = 1.000000
-----------------------------------------
Training error: 0.000494
Training Time:62.718750s

Testing error: 0.130082
Testing Time:12.750000s
```


The first 100 images of NORB training set, and some of their feature maps:


- Original Image:

![Original Image](http://img.blog.csdn.net/20160311220420363)


- Convolutional Feature Maps:

![Convolutional Feature Maps](http://img.blog.csdn.net/20160311220603023)



- Pooling Feature Maps:

![Pooling Feature Maps](http://img.blog.csdn.net/20160311220640337)





