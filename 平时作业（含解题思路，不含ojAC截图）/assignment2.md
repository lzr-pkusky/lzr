# Assignment #2: 编程练习

Updated 0953 GMT+8 Feb 24, 2024

2024 spring, Complied by ==卢卓然 生命科学学院==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:

- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

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

### 27653: Fraction类

http://cs101.openjudge.cn/2024sp_routine/27653/



思路：首先定义__init__函数，包括两个属性分子和分母。然后定义str函数，以进行print。在加法模块，首先用分数的加法法则得到新的分子分母，然后进行约分。具体实现见代码。



##### 代码

```python
class Fraction:
    def __init__(self,up,down):
        self.fenzi=up
        self.fenmu=down
    def __str__(self):
        return str(self.fenzi)+'/'+str(self.fenmu)
    def plus(self,other):
        up1,up2,down1,down2=self.fenzi,other.fenzi,self.fenmu,other.fenmu
        ups=up1*down2+up2*down1
        downs=down1*down2
        a=ups
        for k in range(2,a+1):
            if k<=ups:
                while ups%k==0 and downs%k==0:
                    ups//=k
                    downs//=k
        return Fraction(ups,downs)
u1,d1,u2,d2=map(int,input().split())
print(Fraction.plus(Fraction(u1,d1),Fraction(u2,d2)))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240227162007978](/Users/luzhuoran/Desktop/数算/作业/作业2/image-20240227162007978-9022012.png)



### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110



思路：

计算这n种糖果单位重量的价值，并以此为依据排序。先拿单位价值最大的糖果，然后以此类推，最后按照格式输出。

##### 代码

```python
n,W=map(int,input().split())
candys=[]
for _ in range(n):
    v,w=map(int,input().split())
    candys.append((v,w))
'''计算这n种糖果单位重量的价值,并以此为依据排序'''
candys.sort(key=lambda x: x[0]/x[1],reverse=True)
totalvalue=0
time=0
for k in range(n):
    if candys[k][1]<=W:
        time+=1
        totalvalue+=candys[k][0]
        W-=candys[k][1]
        if W==0:
            print(("{:.1f}".format(totalvalue)))
            break
        elif k==n-1 and W>0:
            print(("{:.1f}".format(totalvalue)))
            break
    else:
        totalvalue+=((candys[k][0])/(candys[k][1]))*W
        print(("{:.1f}".format(totalvalue)))
        break
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240227162320401](/Users/luzhuoran/Desktop/数算/作业/作业2/image-20240227162320401-9022201.png)



### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/



思路：先读取技能，然后按照时间进行排序，然后考察t时刻，若t时刻技能多于m个，则取前m个，否则就全取。然后求和算总伤害。



##### 代码

```python
N=int(input())
output=[]
for _ in range(N):
    n,m,hp=map(int,input().split())
    #读取技能
    skills={}#skills={time:[skill1,skill2,skill3]} for example
    for x in range(n):
        time,effect=map(int,input().split())
        if time not in skills:
            skills[time]=[effect]
        else:
            skills[time].append(effect)
    sorted_skills=dict(sorted(skills.items(),key=lambda x:x[0]))
    for t,skill in sorted_skills.items():
        hp-=sum(sorted(skill,reverse=True)[:m] if len(skill)>=m else skill)
        #若t时刻技能多于m个，则取前m个，否则就全取。然后求和算总伤害。
        if hp<=0:
            break
    output.append('alive' if hp>0 else t)
for o in output:
    print(o)     
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240227162847882](/Users/luzhuoran/Desktop/数算/作业/作业2/image-20240227162847882-9022530.png)



### 230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B



思路：首先打印出质数表。然后验证这个数的开平方是否是质数。



##### 代码

```python
primes=[True for _ in range(10**6+1)]
primes[1]=False
primes[10**6]=False
for i in range(2,10**6):
    if primes[i]==True:
        for j in range(i+i,10**6,i):
            primes[j]=False
#Tprimes=[x**2 for x in primes]
result=[]
input()
l=[int(y) for y in input().split()]
for num in l:
    '''
    if (num>=6 and num%2==0) or num==1:
        result.append('NO')
        continue
        '''
    sqrtnum=num**0.5
    if (sqrtnum)-int(sqrtnum)>0:
        result.append('NO')
        continue
    if primes[int(sqrtnum)]:
        result.append('YES')
    else:
        result.append('NO')
for r in result:
    print(r)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240227163024161](/Users/luzhuoran/Desktop/数算/作业/作业2/image-20240227163024161-9022625.png)



### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A



思路：

严格来讲，并不属于双指针做法。当totalsum能被3整除是，必须至少剔除掉一个不能被3整除的元素才行。而这种剔除可以从左往右也可以从右往左。于是比较这两种方向哪一种剔除的元素更少，就选那种。

##### 代码

```python
def find_longest_subarray(test_cases):
    results = 0
    n, x, arr=test_cases[0],test_cases[1],test_cases[2]
    total_sum = sum(arr)
    if total_sum % x != 0:
        results=n
        return results
    left, right = 0, n - 1
    while left < n and arr[left] % x == 0:
            left += 1
    while right >= 0 and arr[right] % x == 0:
            right -= 1
    if left == n:
            results=-1
    else:
            results=(n - min(left + 1, n - right))
    return results

case=int(input())
output=[]
for _ in range(case):
    n,x=map(int,input().split())
    list=[int(z) for z in input().split()]
    testcase=[n,x,list]
    output.append(find_longest_subarray(testcase))
for o in output:
    print(o)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240227171309847](/Users/luzhuoran/Desktop/数算/作业/作业2/image-20240227171309847-9025191.png)



### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



思路：本题基础是230B.T-primes，判断是否为t-primes然后将符合要求的结果储存起来进行计算。



##### 代码

```python
#(1≤Xi≤10^8)内素数
primetable=[True]*(10**4+1)
primetable[0]=primetable[1]=False
for i in range(2,10**2+1):
    if primetable[i]:
        for j in range(i*i,10**4+1,i):
            primetable[j]=False
m,n=map(int,input().split())
result=[]
for _ in range(m):
    grades=[int(x) for x in input().split()]
    r=0
    count=0
    for g in grades:
        gg=g**0.5
        count+=1
        if gg==int(gg) and primetable[int(gg)]:
            r+=g
    result.append(format(r/count,'.2f') if r>0 else 0)
for re in result:
    print(re)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240227165310010](/Users/luzhuoran/Desktop/数算/作业/作业2/image-20240227165310010-9023991.png)



## 2. 学习总结和收获

本次作业练习了数算中的python编程，27653练习了基础类的书写，18182、04110是数据的处理，230B和18176应用了质数筛法。1364A. XXXXX，这道题思维难度还比较大。

