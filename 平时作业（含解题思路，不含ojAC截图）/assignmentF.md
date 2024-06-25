# Assignment #F: All-Killed 满分

Updated 1844 GMT+8 May 20, 2024

2024 spring, Complied by ==卢卓然 生命科学学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS 

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)

C/C++编程环境：



## 1. 题目

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



思路：

**知识点：二叉树、bfs**。按层遍历，提取出每层最后一个元素即可。

代码

```python
n=int(input())
#建树
treelist=[None]*(n+1)
for i in range(1,n+1):
    a,b=map(int,input().split())
    treelist[i]=(a,b)
#bfs
from collections import deque
q=deque()
q.append(1)
size=1
ans=''
while q:
    for k in range(size):
        tmp=q.popleft()
        if k==size-1:
            ans=ans+str(tmp)+' '
        m,n=treelist[tmp][0],treelist[tmp][1]
        if m!=-1:
            q.append(m)
        if n!=-1:
            q.append(n)
    size=len(q)
print(ans)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240528153917851](/Users/luzhuoran/Desktop/数算/F/image-20240528153917851-6881962.png)



### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



思路：

单调栈精髓：**及时去掉无用数据，保证栈中元素有序**

代码

```python
# 从右往左算，栈中记录的是已经找到答案的数
n=int(input())
nums=[int(y) for y in input().split()]#数字数据
ans=[0]*n#f结果储存
st=[]#创建空栈,栈中储存的事数字在nums中的索引
for i in range(n-1,-1,-1):#倒着遍历
    t=nums[i]
    while st and t>=nums[st[-1]]:
        #元素大于等于栈顶元素，该栈顶元素一定不是f(t),也不是t前面元素的f值，所以没用了
        st.pop()
    if st:
        #元素小于栈顶元素，说明栈顶元素就是f(t)
        ans[i]=st[-1]+1#下标加上一，满足题意
    st.append(i)
print(' '.join(map(str,ans)))
```

```python
#从左往右算，栈中记录的是没有找到答案的数，更体现单调栈特点,每出现比栈顶大的数就更新栈顶的ans值，每出现比栈顶小的数就压入栈中。
from collections import deque
n=int(input())
ans=[0]*n
nums=[int(X) for X in input().split()]
st=deque()
for i ,t in enumerate(nums):
    while st and t>nums[st[-1]]:
        j=st.pop()
        ans[j]=i+1
    st.append(i)
print(*ans)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240528164810118](/Users/luzhuoran/Desktop/数算/F/image-20240528164810118-6886091.png)



### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



思路：来源熊江凯题解，仔细阅读了他的思路，感悟颇多。



代码

```python
cur=''
for _ in range(int(input())):
    try:n,m=list(map(int,input().split()))
    except:break
    vis,edge=[0]*(n+5),[[]for _ in range(n+1)]
    for _ in range(m):
        a,b=list(map(int,input().split()))
        edge[a].append(b)
    def dfs(x,f):
        global vis,edge
        vis[x],aa=1,0
        for i in edge[x]:
            if vis[i]==0 and edge[i]:aa=dfs(i,f+[i])
            elif i in f:return 1
            if aa:return 1
        return 0
    ans=0
    for i in range(1,n+1):
        if edge[i] and vis[i]==0:
            ans=dfs(i,[i])
            if ans:break
    cur+=(('Yes'if ans else'No')+'\n')
print(cur)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240528234925626](/Users/luzhuoran/Desktop/数算/F/image-20240528234925626-6911366.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：

在计概时做过这道题目，非常好、有难度。

代码

```python
n,m=list(map(int,input().split()))
out=[int(input())for _ in range(n)]
l,r,ans=max(out),sum(out),0
def check(x):
    global out,m,n
    res,i,tot=m,0,0
    while res>0 and i<n:
        if tot+out[i]<=x:tot+=out[i]
        else:
            res-=1
            if res==0:return 0
            tot=out[i]
        i+=1
    if i==n:return 1
    return 0
while l<=r:
    mid=(l+r)//2
    if check(mid):
        ans=mid
        r=mid-1
    else:l=mid+1
print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240528235042285](/Users/luzhuoran/Desktop/数算/F/image-20240528235042285-6911443.png)



### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



思路：有权最短路径——dijkstra



代码

```python
from heapq import heappush as hu,heappop as hp
k,n,r=[int(input())for _ in range(3)]
edge,vis=[[]for _ in range(n+1)],[100000]*(n+1)
for _ in range(r):
    x,y,z,w=map(int,input().split())
    edge[x].append((y,z,w))
q,ans=[],-1
hu(q,(0,0,1))
while q:
    l,c,x=hp(q)
    if x==n:
        ans=l
        break
    vis[x]=c
    for y,z,w in edge[x]:
        if c+w<vis[y] and c+w<=k:hu(q,(l+z,c+w,y))
print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240528235217315](/Users/luzhuoran/Desktop/数算/F/image-20240528235217315-6911538.png)



### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：

并查集，这对我来说是个难点，一直没有特别理解，还需要加强。

代码

```python
n,k=list(map(int,input().split()))
f,ans=[i for i in range(n*3+1)],0
def find_x(x):
    global f
    if f[x]==x:return x
    f[x]=find_x(f[x])
    return f[x]
def union(x,y):
    global f
    fx,fy=find_x(x),find_x(y)
    f[fx]=fy
for _ in range(k):
    d,x,y=list(map(int,input().split()))
    if (x>n or y>n)or(d==2 and x==y):
        ans+=1
        continue
    fx1,fx2,fx3,fy=find_x(x),find_x(n+x),find_x(n*2+x),find_x(y)
    if (d==1 and (fx2==fy or fx3==fy)) or (d==2 and (fx1==fy or fx3==fy)):
        ans+=1
        continue
    if d==1:
        union(x,y)
        union(x+n,y+n)
        union(x+2*n,y+2*n)
    else:
        union(x,2*n+y)
        union(y,n+x)
        union(x+2*n,y+n)
print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240528235343788](/Users/luzhuoran/Desktop/数算/F/image-20240528235343788-6911624.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

本次作业的知识点清单。现在我要把所有知识点都研究透彻，以应对考试！本次作业质量很高，值得反复回味。

| problems                    | tags             |
| --------------------------- | ---------------- |
| 22485: 升空的焰火，从侧面看 | binary tree, bfs |
| 28203:【模板】单调栈        | monotonous stack |
| 09202: 舰队、海域出击！     | topological sort |
| 04135: 月度开销             | binary search    |
| 07735: 道路                 | Dijkstra         |
| 01182: 食物链               | disjoint set     |



