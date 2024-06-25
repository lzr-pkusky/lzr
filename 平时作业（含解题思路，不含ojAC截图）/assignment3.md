# Assignment #3: March月考

Updated 1537 GMT+8 March 6, 2024

2024 spring, Complied by ==卢卓然 生命科学学院==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS

Python编程环境：PyCharm

C/C++编程环境：



## 1. 题目

**02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/



思路：最长下降子序列，dp做法。dp[i]指以l[i]结尾的最长下降子序列的长度，初始化为1因为当这个数字前面的都比他小时，dp[i]应该是1，也就是说，前i个中的最长下降子序列至少为1（也就是只包含第i个数）。状态转移方程，对于前面的l[j]如果l[j]比l[i]大/相等，那么可以拼接，所以dp[j]+1。如此这般，找出最大的可能的dp[i]。



##### 代码

```python
#最长下降子序列
n=int(input())
l=[int(x) for x in input().split()]
dp=[1]*n
#dp[i]指以l[i]结尾的最长下降子序列的长度
dp[0]=1
for i in range(1,n):
    a=l[i]
    for j in range(i):
        if l[j]>=a:
            dp[i]=max(dp[i],dp[j]+1)
print(max(dp))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240310162255140](/Users/luzhuoran/Desktop/数算/作业/作业3‘/image-20240310162255140-0058976.png)



**04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147



思路：先规定一次移动的print输出格式，然后写递归函数hanno（）。当n=1是，是一次移动，是递归的基本情况。递归的框架就是先把1-N-1从A经过C移动到B，再把N从A移动到C，最后把1-N-1从B经过A移动到C。



##### 代码

```python
#先把1-N-1从A经过C移动到B，再把N从A移动到C，最后把1-N-1从B经过A移动到C
def move(columnnumber,start,end):#一次移动
    print(str(columnnumber)+':'+f'{start}->{end}')
def hanno(n,start,way,end):#把1-n从start经过way移动到end
    if n==1:
        move(1,start,end)
        return
    hanno(n-1,start,end,way)
    move(n,start,end)
    hanno(n-1,way,start,end)
N,s,w,e=input().split()
N=int(N)
hanno(N,s,w,e)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240310153025281](/Users/luzhuoran/Desktop/数算/作业/作业3‘/image-20240310153025281-0055826.png)



**03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253



思路：思路就是模拟，一个一个从列表里删去人，然后找一个人的索引（用取余数实一个圈的效果）再删除。

要注意`point=((point)%length)` 这一步很重要。



##### 代码

```python
result=[]
while True:
    n,p,m=map(int,input().split())
    if n==p==m==0:
        break
    l=[i for i in range(1,n+1)]
    output=[]
    length=n
    point=(p-1+m)%n-1
    output.append(l[point])
    while length>1:
        l.remove(l[point])
        length-=1
        point+=(m-1)
        point=((point)%length)
        output.append(l[point])
    result.append(','.join(map(str,output)))
for r in result:
    print(r)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240310135952085](/Users/luzhuoran/Desktop/数算/作业/作业3‘/image-20240310135952085-0050395.png)



**21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554



思路：根据测试数据用例，猜测是对数组进行排序，排序后的结果即为所求顺序。再计算平均排队时间。



##### 代码

```python
n=int(input())
l=[(index+1,int(x)) for index,x in enumerate(input().split())]
s=sorted(l,key=lambda x:x[1])
output=[str(a[0]) for a in s]
print(' '.join(output))
sum=0
for i in range(n-1):
    sum+=(s[i][1])*(n-1-i)
print('{:.2f}'.format(sum/n))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240310163815059](/Users/luzhuoran/Desktop/数算/作业/作业3‘/image-20240310163815059-0059896.png)



**19963:买学区房**

http://cs101.openjudge.cn/practice/19963



思路：中位数：偶数个数据：n//2个和n//2+1个的平均数；奇数个数据：第n//2+1个数据。剩下都是数据处理了。



##### 代码

```python
 #中位数：偶数个数据：n//2个和n//2+1个的平均数；奇数个数据：第n//2+1个数据
n=int(input())
distance=[sum([int(x) for x in strtuple.lstrip('(').rstrip(')').split(',')]) for strtuple in input().strip().split()]
value=[int(y) for y in input().strip().split()]
performance=[d/v for d,v in zip(distance,value)]
#print(performance)
sp=sorted(performance)
sv=sorted(value)
if n%2==0:
    middle_perf=(sp[n//2-1]+sp[n//2])/2
    middle_valu=(sv[n//2-1]+sv[n//2])/2
else:
    middle_perf=sp[n//2]
    middle_valu=sv[n//2]
r=0
for h in range(n):
    if performance[h]>middle_perf and value[h]<middle_valu:
        r+=1
print(r)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240310163929265](/Users/luzhuoran/Desktop/数算/作业/作业3‘/image-20240310163929265-0059970.png)



**27300: 模型整理**

http://cs101.openjudge.cn/practice/27300



思路：主要是集合字典的使用，以及数据处理的功力。排序时可以指定标准key=lambda x，x就代表了序列的一个元素。



##### 代码

```python
s=set()
dM=dict()
dB=dict()
n=int(input())
for _ in range(n):
    name,data=input().split('-')
    if name not in s:
        s.add(name)
        dM[name]=[]
        dB[name]=[]
    if data[-1]=="M":
        dM[name].append(data)
    elif data[-1]=="B":
        dB[name].append(data)
for Name in sorted(list(s)):
    newlist=sorted(dM[Name],key=lambda x : int(x[:-1]) if '.' not in x else float(x[:-1]))+sorted(dB[Name],key=lambda x : int(x[:-1]) if '.' not in x else float(x[:-1]))
    print(Name+': '+', '.join(newlist))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240310170113983](/Users/luzhuoran/Desktop/数算/作业/作业3‘/image-20240310170113983-0061274-0061276.png)



## 2. 学习总结和收获



本次作业有一道简单递归，一道简单dp，一道模拟，三道处理数据的题目。

汉诺塔作为经典递归，充分展现了递归的一般做法：先假定函数hanno能够完成任务——把1-N盘子从start经过way移动到end，然后将这个大问题分解成三个步骤，第一步先把1-N-1从A经过C移动到B（超级操作），再把N从A移动到C（微操作），最后把1-N-1从B经过A移动到C（超级操作）。对于两个超级操作，完全符合hanno(n-1,A,C,B) 和hanno(n-1,B,A,C)的定义。而递归若想顺利的运行起来，需要定义递归的基本情况，递归的基本情况就是一次微操作。当n=1时，将盘子1从start经过way移动到end的hanno操作，就相当于将盘子1从start移动到end的move微操作。定义了基本情况，递归就可以顺利运行下去了。

对于约瑟夫问题，我采用了直接模拟的方法，oj并没有算我超时，并没有采取其他数据结构。



