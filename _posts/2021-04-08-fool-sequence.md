---
author: aaeria
---

A fool's number is a number with digits comprised only of '69' and '420'. This program can find the nth fool's number, when sorted in increasing order.

[Program]({{'/widgets/foolsequence.html' | relative_url}})

## Growth

A fools number can be formed by appending 69 or 420 to another smaller fool number. 
As such, if $$d_i$$ is the number of fool numbers with $$i>3$$ digits, then $$d_i=d_{i-2}+d_{i-3}$$ (we can define $$d_3=1,d_2=1,d_1=0$$).

The asymtotic growth of $$d_n$$ can be found with function $$x^3=1+x$$, which gives $$x≈1.3247, d_n≈1.3247^n$$

Then the number of digits with $$n$$ or fewer digits is $$s_n=\sum_{i=1}^n d_i=\theta(1.3247^n)$$

Of course, the $$s_n$$th number has $$n$$ digits so it is approximately $$10^n$$. The $$n$$th fool number is $$\theta(10^{\log_{1.3247}n })=\theta(n^{8.18})$$

![Fools sequence plot]({{'/assets/images/Fool_seq_Figure_1.png' | relative_url}})