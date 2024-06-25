# Assignment #1: 拉齐大家Python水平

Updated 0940 GMT+8 Feb 19, 2024

2024 spring, Complied by ==卢卓然 生命科学学院==



**说明：**

1）数算课程的先修课是计概，由于计概学习中可能使用了不同的编程语言，而数算课程要求Python语言，因此第一周作业练习Python编程。如果有同学坚持使用C/C++，也可以，但是建议也要会Python语言。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS

Python编程环境：PyCharm

C/C++编程环境：



## 1. 题目

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/



思路：递归。为了防止爆栈，使用了lru_cache进行缓存。



##### 代码

```python
from functools import lru_cache
@lru_cache(maxsize=None)
def f(n):
    if n==0:
        return 0
    elif n==1 or n==2:
        return 1
    else:
        return f(n-3)+f(n-2)+f(n-1)

print(f(int(input())))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240219131734946](/Users/luzhuoran/Desktop/数算/作业/作业1/image-20240219131734946.png)



### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A



思路：

依次寻找hello五个字符，用flag保证前后顺序。

##### 代码

```python
s=input()
h=len(s)
flag=0
for i in range(h):
    letter=s[i]
    if flag==0 and letter=='h':
        flag+=1
    elif flag==1 and letter=='e':
        flag+=1
    elif flag==2 and letter=='l':
        flag+=1
    elif flag==3 and letter=='l':
        flag+=1
    elif flag==4 and letter=='o':
        flag+=1
if flag==5:
    print('YES')
else:
    print('NO')


```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240219133051493](/Users/luzhuoran/Desktop/数算/作业/作业1/image-20240219133051493-8320652.png)



### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A



思路：

先用一个元组（最好用集合）储存所有元音，然后遍历这个字符串，并做替换，组成新的字符串。

##### 代码

```python
vowels=('a','e','i','o','u','y')
word=list(input().lower())
newword='.'
for letter in word:
    if letter not in vowels:
        newword=newword+letter+'.'
print(newword[0:-1])
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240219133551993](/Users/luzhuoran/Desktop/数算/作业/作业1/image-20240219133551993-8320955.png)



### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/



思路：欧拉筛打表，然后查找第一个最小的质数，再检查n-k是否为质数，若是，直接结束循环输出结果。



##### 代码

```python
n=int(input())
primetable=[True]*(n+1)
primetable[0]=primetable[1]=False
for i in range(2,int(n**0.5)+1):
    if primetable[i]:
        for j in range(i*i,n+1,i):
            primetable[j]=False
for k in range(n):
    if primetable[k]:
        if primetable[n-k]:
            break
print(f'{k} {n-k}')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==



![image-20240219134248737](/Users/luzhuoran/Desktop/数算/作业/作业1/image-20240219134248737-8321369.png)

### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



思路：用两次split函数预处理数据，然后遍历所有数据，若系数非0，则更新最大次数。



##### 代码

```python
strings=input().split('+')
tuple_strings=[(s.split('n^')) for s in strings]
maximum=0
for string in tuple_strings:
    if string[0]!='0':
        maximum=max(maximum,int(string[1]))
print('n^'+str(maximum))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240219134355544](/Users/luzhuoran/Desktop/数算/作业/作业1/image-20240219134355544-8321436.png)



### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/



思路：

用字典储存数字和他出现的次数。然后遍历字典，找到次数最多的数字。最后排序输出。

##### 代码

```python
nums=[int(z) for z in input().split()]
dic=dict()
s=set()
for num in nums:
    if num in s:
        dic[num]+=1
    else:
        s.add(num)
        dic[num]=1
maxnum=[]
maxcount=0
for number,count in dic.items():
    if count>maxcount:
        maxnum=[number]
        maxcount=count
    elif count==maxcount:
        maxnum.append(number)
print(' '.join([str(u) for u in sorted(maxnum)]))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==



![image-20240219141146826](/Users/luzhuoran/Desktop/数算/作业/作业1/image-20240219141146826.png)

## 2. 学习总结和收获

本次作业，专注于数据结构与语法的运用。涵盖了基本递归、字符串、列表、字典、集合等知识点。前5道题目在计算概论B课程中，已经练习过，因此完成该作业所用时间较少。



