---
title: 加密算法|RSA
date: 2020-09-10
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- Cybersecurity|网络安全
mathjax: true
toc: true
comments: 基础加密算法分析

---

# RSA

> 本文为非对称加密算法 RSA 定义，数学基础，RSA 密码体制基础，相关算法笔记与代码的整理。偏向于教科书内容，现实中的 RSA 实现仍需要在教科书版的 RSA 上打补丁升级。

## 单向函数定义

+ 假定 $n=p q(p, q$ 为不同的大素数 ), b 为正整数, 定义

$$
f: \mathbb{Z}_{n} \rightarrow \mathbb{Z}_{n} f(x)=x^{b} \bmod n
$$

+ 陷门：大数 n 的因式分解
  若已知 $n \theta$ 的因式分解，即$n=p q,$ 则 $\varphi(n)=(p-1)(q-1)$ 若 $\mathrm{gcd}(b, \varphi(n))=1,$ 且 $a b \equiv 1 \bmod \varphi(n)$
  
  <!--more-->

## 数学基础

+ Euler 定理 设 $(a, n)=1,$ 则有 $a^{\phi (n)} \equiv 1 \bmod n$
  特别地，当 p 为素数时，对任意 $a$ 有 $a^{p} \equiv a \bmod p,$ 称为 $F e r m a t$ 小定理。

+  **正整数 $n>1, \forall a$ st $0 \leq a<n$ 和 $k>0,$ 均 有 $a^{1+k\phi (n)} \equiv a \bmod n$** 
  当且仅当 n 有分解式 $n=\prod p_{i}$ （其中 $p_{i}$ 为不同素数 )

  证明：

  先 证 明 充 分 性，即 当 $n=\prod_{i=1}^{r} p_{i}$ 时 $, \quad a^{1+k_{\theta}(n)}=a \bmod n$
  1 .若$ n|a$ ,  有 $a^{1+k p(n)} \equiv 0 \equiv a \bmod n$ 成立 
  2.若 $a$ 非 $0 \quad n=\prod_{i=1}^{r} p_{i}, \therefore \varphi(n)=\prod_{i=1}^{r} \varphi\left(p_{i}\right)$

  $$
 \forall p_{j} 1 \leq j \leq r,
  
 1 \text { )if } p_{j} \mid a, \text { then } a^{1+k_{\varphi}(n)} \equiv 0 \equiv a \bmod p_{j}
  $$

  ​			2）若 $a \neq 0\ mod\$  $p_{j},$ 则 $\left(a, p_{j}\right)=1,$ 根 据 欧 拉 定 理 有 $a^{\phi\left(p_{j}\right)}=1\ mod\ p_{j}$
  ​			则 $a^{1+k e(n)} \bmod p_{j}=a^{1+k \prod_{i=1}^{n} \varphi\left(p_{i}\right) \quad}\bmod p_{j}=a \bmod p_{j}$
  ​			综合 1 )和 2 ) 有 

  $a^{1+k_{9}(n)}=a \bmod p_{j}$
  根据 $\left\{\begin{array}{c}a^{1+k_{9}(n)}=a \bmod p_{1} \\ \ldots \\ a^{1+k_{9}(n)}=a \bmod p_{r}\end{array}\right.$ 且 $n=\prod_{i=1}^{r} p_{i}, \therefore$ 有 $a^{1+k q(n)}=a \bmod$
  综合 1 和 2，充分性得证。

### RSA 密码体制

设 $n=p q,$ 其中 $p, q$ 素数，设 $\mathrm{P}=\mathrm{C}=\mathbb{Z}_{n},$ 且定义

$$
\mathrm{K}=\{(n, p, q, e, d): e d=1 \bmod \varphi(\mathrm{n})\}
$$

对于 $k=(n, p, q, e, d),$ 定义

$$
e_{k}(x)=x^{e} \bmod n
$$

和

$$
d_{k}(y)=y^{d} \bmod n
$$

$\left(x y \in \mathbb{Z}_{n}\right)(n, e)$ 为公钥 $,(n, d)$ 为私钥
根据前面的结论，很突易证明 $d_{k}\left(e_{k}(x)\right)=x^{e d}=x \bmod n$

$a^{1+k\phi (n)} \equiv a \bmod n, k = 1, ed$ 可以表示为 $1+k\phi(n)$  

注： $p,q,\varphi(n)$ 需要保密
$\left\{\begin{array}{c}n=p q \\ \varphi(\mathrm{n})=(p-1)(q-1)\end{array}\right.$
$\Rightarrow p^{2}-(n-\varphi(n)+1) p+n=0$ 

## 相关算法

### RSA 加密于解密

Bob 发送 m 给 Alice

+ Bob 使用 Alice 公钥 $(n,e)$加密得到 $C = m^e \mod n$

+ Alice 做 $m = C^d \mod n$ 得到明文

### RSA 签名方案

Bob 发送消息给 Alice，发送的内容为 m

+ Bob 使用自己的私钥加密，发送 $(m,H(m) ^d)$ 给 Alice,Alice 收到 $(m,s)$
+ Alice 用 Bob 的公钥做签名验证: 如果$H(m)=s^e$ 接受签名 



### Euclid's Algorithm - GCD

```python
def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)
```

证明:

设 $a = km,b = im, a>b,k>i,(k,i)=1$ 其中 m 是 a 与 b 的最大公约数 , $a - nb = (k-ni)m$ 等价于 $a\%b =(k-ni)m,(k-ni  ) \in Z\$ 且由于求余的性质可得 $(k-ni)<i\ \&\ (k-ni,i)=1:$ , 可见 $m$是$b$与$a\%b$的一个公约数 ,又因为$(k-ni,i)=1$,所以$m$为两者最大公约数, $gcd(a,b) = gcd(b,a\%b)$ 

### Extended Euclidean Algorithm

```python
def ext_euclidean(a, b): # to obtain x s.t. ax=1(mod b), if a > b, please do a %= b before calling this function
    if a == 1:
        return 1
    else:
        return (1 + (a - ext_euclidean(b % a, a)) * b) // a
```

证明:

notation: $p = mmi(a,b):a\times p \equiv 1\bmod b$  

$p\times a + q\times b = 1\ where\ p,q\in \mathbb{Z}$

let $b = s\times a + r, \therefore (p+q\times s)\times a+q\times r = 1$

$\therefore q = mmi(r,a)$

带入上式得: $p = \frac{1-mmi(r,a)\times b}{a}$ 

### Generate Prime 

```python
import rabinmiller
import re


primelength = raw_input('please input the length of the prime:')  # input should less then 1024
filename = raw_input('please input filename to store the prime:')
file = open(filename, "w")
n1 = rabinmiller.generateLargePrime(int(primelength))
file.write(str(n1))
file.close()
```

```python
# Primality Testing with the Rabin-Miller Algorithm
# http://inventwithpython.com/hacking (BSD Licensed)

import random


def rabinMiller(num):
    s = num - 1
    t = 0
    while s % 2 == 0:
        # keep halving s until it is even (and use t
        # to count how many times we halve s)
        s = s // 2
        t += 1

    for trials in range(5): # try to falsify num's primality 5 times
        a = random.randrange(2, num - 1)
        v = pow(a, s, num)
        if v != 1: # this test does not apply if v is 1.
            i = 0
            while v != (num - 1):
                if i == t - 1:
                    return False
                else:
                    i = i + 1
                    v = (v ** 2) % num
    return True


def isPrime(num):
    # Return True if num is a prime number. This function does a quicker
    # prime number check before calling rabinMiller().

    if (num < 2):
        return False # 0, 1, and negative numbers are not prime

   # About 1/3 of the time we can quickly determine if num is not prime
    # by dividing by the first few dozen prime numbers. This is quicker
    # than rabinMiller(), but unlike rabinMiller() is not guaranteed to
    # prove that a number is prime.
    lowPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

    if num in lowPrimes:
        return True
    for prime in lowPrimes:
        if (num % prime == 0):
            return False
    return rabinMiller(num)


def generateLargePrime(keysize=1024):
    while True:
        num = random.randrange(2 **(keysize-1), 2** (keysize))
        if isPrime(num):
            return num
```

### Generate RSA key

```python
import random
import cryptomath
import re


def inputprime(filepath):
        try:
                Pfile = open(filepath, "r")
        except:
                print("-----file not exists-----")
                return 0
        return int(Pfile.readline())

# input of this program: prime P and prime Q
# Output of this program: \nE, which is relatively prime to (P-1)*(Q-1)\nD, which is the mod inverse of E under mod (P-1)*(Q-1)

Pvalue = inputprime(raw_input("Please input the filename of the first prime P: "))

while Pvalue == 0:
        Pvalue = inputprime(raw_input("Please input the filename of the first prime P: "))

Qvalue = inputprime(raw_input("Please input the filename of the second prime Q: "))

while Qvalue == 0:
        Qvalue = inputprime(raw_input("Please input the filename of the second prime Q: "))

        keySize = (len(bin(Pvalue)[2:]) + len(bin(Qvalue)[2:])) / 2

while True:
        Evalue = random.randrange(2  **(keySize - 1), 2**  (keySize))
        if cryptomath.gcd(Evalue, (Pvalue - 1) * (Qvalue - 1)) == 1:
                break

Dvalue = cryptomath.findModInverse(Evalue, (Pvalue - 1) * (Qvalue - 1))
filename = raw_input('please input a filename to store all the results:')

while not re.match('^[0-9a-zA-Z]', filename):
        filename = raw_input('please input a filename to store all the results:')

file = open(filename, "w")
file.write("P: " + str(Pvalue))
file.write("\nQ: " + str(Qvalue))
file.write("\nD: " + str(Dvalue))
file.write("\nE: " + str(Evalue))
file.write("\nP*Q: " + str(Pvalue * Qvalue))
file.write("\n(P-1)*(Q-1): " + str((Pvalue - 1) * (Qvalue - 1)))
file.close()
```



## 相关问题

+ 如果加密的数据为数字, 攻击者做 $C \times 2^e$. 接收者不会意识到被攻击了。
+ 一样的密钥与明文生成一样的密文
+ 加密时间过长

解决方案

+ padding

+ E 通常用 65537，这是个被公众认为可以加快加密过程，并且保留安全性的数字

+ 使用密钥长度: 1024 bits (309 digits), 现多用 2048 bits

+ 使用 DES/AES 加密 M，然后用 RSA 加密其密钥

  

## 量子计算机的威胁: 舒尔算法 - Shor 

+ time complexity - $O(\log n)$  

+ 持续更新中
