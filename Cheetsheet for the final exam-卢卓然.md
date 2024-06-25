# Cheetsheet for the final exam

## 语法工具

### 埃氏筛法，得到质数表

```python
def judge(number):
    nlist = list(range(1,number+1))
    nlist[0] = 0
    k = 2
    while k * k <= number:
        if nlist[k-1] != 0:
            for i in range(2*k,number+1,k):
                nlist[i-1] = 0
        k += 1
    result = []
    for num in nlist:
        if num != 0:
            result.append(num)
    return result
```

### 二分查找（bisect)

python中直接用bisect包,进行二分查找

```python
import bisect ##导入bisect包

##bisect是一个排序模块，操作对象必须为排好序的列表。
##bisect操作并不改变列表中的元素，仅仅是确认插入元素的位置
##与之对应的insort
lst = [1,3,5,7,9]
s = int(input())
bisect.bisect_left(lst, x, [lo=0, hi=len(a)])   ##[]中表示插入位置的上界和下届
##改成right同理

## 测试序列 a2
>>> a2 = [1, 3, 3, 4, 7]  # 元素从小到大排列，有重复, 不等距
 
# 限定查找范围：[lo=1, hi=3] 
>>> bisect.bisect_left(a2, 0, 1, 3)  # 与 x=0 右侧最近的元素是 1, 其位置 index=0, 但下限 lo=1, 故只能返回位置 index=1
1
3
##如果说 bisect.bisect_left() 是为了在序列 a 中 查找 元素 x 的插入点 (左侧)，那么 bisect.insort_left() 就是在找到插入点的基础上，真正地将元素 x 插入序列 a，从而改变序列 a 同时保持元素顺序。
>>> a12 = [5, 6, 7, 8, 9]
>>> bisect.insort_left(a12, 5.5)
>>> a12
[5, 5.5, 6, 7, 8, 9]

```

## 

### math

gcd包，计算最大公因式

```python
from math import gcd
x = gcd(15,20,25)
print(x)
## 5
```

math.pow(m,n)	计算`m`的`n`次幂。

math.log(m,n)	计算以`n`为底的`m`的对数。

### eval

eval() 是 python 中功能非常强大的一个函数
将字符串当成有效的表达式来求值，并返回计算结果
所谓表达式就是：eval 这个函数会把里面的字符串参数的引号去掉，把中间的内容当成Python的代码，eval 函数会执行这段代码并且返回执行结果
也可以这样来理解：eval() 函数就是实现 list、dict、tuple、与str 之间的转化
————————————————

```python
result = eval("1 + 1")
print(result)  # 2   

result = eval("'+' * 5")
print(result)  # +++++

# 3. 将字符串转换成列表
a = "[1, 2, 3, 4]"
result = type(eval(a))
print(result)  # <class 'list'>

input_number = input("请输入一个加减乘除运算公式：")
print(eval(input_number))
## 1*2 +3
## 5
```

 **for _ in sorted(dic.keys()): ##将字典按keys排序后**

**float('inf') 表示正无穷**

**##注意break只退一层循环**

print(*lst) ##把列表中元素顺序输出

int(str,n)	将字符串`str`转换为`n`进制的整数。

for key,value in dict.items()	遍历字典的键值对。

for index,value in enumerate(list)	枚举列表，提供元素及其索引。

dic.setdefault(key,[]).append(value) 常用在字典中加入元素的方式（如果没有值就建空表，有值就直接添加）

dict.get(key,default) 	从字典中获取键对应的值，如果键不存在，则返回默认值`default`。

list(zip(a,b))	将两个列表元素一一配对，生成元组的列表。



1. `str.lstrip() / str.rstrip()`: 移除字符串左侧/右侧的空白字符。

2. `str.find(sub)`: 返回子字符串`sub`在字符串中首次出现的索引，如果未找到，则返回-1。

3. `str.replace(old, new)`: 将字符串中的`old`子字符串替换为`new`。

4. `str.isalpha() / str.isdigit() / str.isalnum()`: 检查字符串是否全部由字母/数字/字母和数字组成。

5. .`str.title()`：每个单词首字母大写。


## 算法

### 一、栈与队列

##### 出栈序列

```python
#22068:合法出栈序列
from collections import deque
x=input()
while 1:
    try:
        s=input()
        i=0
        if len(x)!=len(s):
            print('NO')
            continue
        a=[];q=deque(s)
        i=0
        while 1:
            try:
                if a and a[-1]==q[0]:
                    a.pop()
                    q.popleft()
                else:
                    a.append(x[i])
                    i+=1
            except:
                break
        if a:
            print('NO')
        else:
            print('YES')
    except EOFError:
        break
```

```python
#04077:出栈序列统计（卡特兰数）
from math import comb
n=int(input())
print(int(comb(2*n, n)/(n+1)))
```

##### 中序表达式转后序表达式

```python
operators = ['+', '-', '*', '/']
def is_num(s):
    for i in operators + ['(', ')']:
        if i in s:
            return False
    return True
def process(raw_input):
    # convert the raw input into separated sequence
    temp, ans = '', []
    for i in raw_input.strip():
        if is_num(i):
            temp += i
        else:
            if temp:
                ans.append(temp)
            ans.append(i)
            temp = ''
    if temp:
        ans.append(temp)
    return ans
def infix_to_postfix(expression):
    # Shunting Yard Algorithm
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output_stack, op_stack = [], []
    for i in expression:
        if is_num(i):
            output_stack.append(i)
        elif i == '(':
            op_stack.append(i)
        elif i == ')':
            while op_stack[-1] != '(':
                output_stack.append(op_stack.pop())
            op_stack.pop()
        else:
            while op_stack and op_stack[-1] in operators and precedence[i] <= precedence[op_stack[-1]]:
                output_stack.append(op_stack.pop())
            op_stack.append(i)
    if op_stack:
        output_stack += op_stack[::-1]
    return output_stack
n = int(input())
for i in range(n):
    tokenized = process(input())
    print(' '.join(infix_to_postfix(tokenized)))
```

##### 后序表达式求值

```python
#赵语涵2300012254
def ope(a,b,o):
    if o=='+':
        return a+b
    elif o=='-':
        return a-b
    elif o=='*':
        return a*b
    elif o=='/':
        return a/b

def calcu(data):
    ind,nums = 0,[]
    while ind < len(data):
        try:
            nums.append(float(data[ind]))
        except:
            b,a = nums.pop(),nums.pop()
            nums.append(ope(a,b,data[ind]))
        ind += 1
    return nums
n = int(input())
for i in range(n):
    x = calcu(list(input().split()))[0]
    print('%.2f'%x)
```

##### 单调栈

精髓：**及时去掉无用数据，保证栈中元素有序**

 f(i) 代表数列中第 i 个元素之后第一个大于 ai 的元素的下标

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
#ans 数组能够反映每个元素之后第一个大于它的数的索引
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

```python
#ans 数组能够反映每个元素之后第一个小于它的数的索引
from collections import deque
n = int(input())
ans = [0] * n
nums = [int(x) for x in input().split()]
st = deque()
for i, t in enumerate(nums):
    while st and t < nums[st[-1]]:
        j = st.pop()
        ans[j] = i + 1
    st.append(i)
print(*ans)
```



##### 辅助栈



单调队列OJ26978:滑动窗口最大值

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        q = collections.deque()
        for i in range(k):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)

        ans = [nums[q[0]]]
        for i in range(k, n):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)
            while q[0] <= i - k:
                q.popleft()
            ans.append(nums[q[0]])

        return ans
```



OJ04137:最小新整数

http://cs101.openjudge.cn/practice/04137/

```py
def removeKDigits(num, k):
    stack = []
    for digit in num:
        while k and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    while k:
        stack.pop()
        k -= 1
    return int(''.join(stack))
t = int(input())
results = []
for _ in range(t):
    n, k = input().split()
    results.append(removeKDigits(n, int(k)))
for result in results:
    print(result)
```

OJ27205:护林员盖房子 加强版

http://cs101.openjudge.cn/2024sp_routine/27205/

题解：https://zhuanlan.zhihu.com/p/162834671

```py
def maximalRectangle(matrix) -> int:
    if (rows := len(matrix)) == 0:
        return 0

    cols = len(matrix[0])
    # 存储每一层的高度
    height = [0 for _ in range(cols + 1)]
    res = 0

    for i in range(rows):  # 遍历以哪一层作为底层
        stack = [-1]
        for j in range(cols + 1):
            # 计算j位置的高度，如果遇到1则置为0，否则递增
            h = 0 if j == cols or matrix[i][j] == '1' else height[j] + 1
            height[j] = h
            # 单调栈维护长度
            while len(stack) > 1 and h < height[stack[-1]]:
                res = max(res, (j - stack[-2] - 1) * height[stack[-1]])
                stack.pop()
            stack.append(j)
    return res


rows, _ = map(int, input().split())
a = [input().split() for _ in range(rows)]

print(maximalRectangle(a))
```

辅助栈

OJ22067:快速堆猪

http://cs101.openjudge.cn/2024sp_routine/22067/

```python
a = []
m = []

while True:
    try:
        s = input().split()
    
        if s[0] == "pop":
            if a:
                a.pop()
                if m:
                    m.pop()
        elif s[0] == "min":
            if m:
                print(m[-1])
        else:
            h = int(s[1])
            a.append(h)
            if not m:
                m.append(h)
            else:
                k = m[-1]
                m.append(min(k, h))
    except EOFError:
        break
```

## 

### 二、图

#### 1.骑士周游

它涉及到在一个棋盘上移动一个骑士（象棋中的马），使其按照骑士的走法（走“日”字）访问棋盘上的每一个方格恰好一次。如果骑士能够回到起始位置，这样的路径称为“封闭巡游”，否则称为“开放巡游”。

```python
def getNeighbor(pos):
    x, y = pos
    return [(x+dx, y+dy) for dx, dy in direc if 0<=x+dx<n and 0<=y+dy<n and graph[x+dx][y+dy]]

def dfs(x, y, num):
    graph[x][y] = 0
    if num == n*n:
        return True
    for x2, y2 in sorted(getNeighbor((x, y)), key=lambda p:len(getNeighbor(p))):
        if graph[x2][y2]:
            if dfs(x2, y2, num+1):
                return True
            graph[x2][y2] = 1
'在这里，getNeighbor 函数首先获取当前位置 (x, y) 的所有合法邻居位置。然后，使用 sorted 函数和一个 lambda 函数作为排序的键（key），根据每个邻居位置的可访问邻居数（即未访问的相邻方格数）来对它们进行排序。这样，每次迭代都会选择具有最少未访问邻居数的位置作为下一步移动，这正是 Warnsdorff’s Rule 的核心思想'
n = int(input())
x, y = map(int, input().split())
direc = [(j*i, k*(3-i)) for i in [1,2] for j in [-1,1] for k in [-1,1]]
graph = [[1]*n for i in range(n)]
ans = 0
print('success' if dfs(x, y, 1) else 'fail')
```

#### 2.拓扑排序/Kahn算法

拓扑排序是对有向无环图（DAG，Directed Acyclic Graph）的顶点进行排序的一种方法，使得对于图中的每条有向边 UV（从顶点 U 指向顶点 V），U 在排序中都出现在 V 之前。拓扑排序不是唯一的，一个有向无环图可能有多个有效的拓扑排序。

拓扑排序常用的算法包括基于 DFS（深度优先搜索）的方法和基于 BFS（广度优先搜索，也称为Kahn算法）的方法。

作用：检测是否有环

1.无向图
使用拓扑排序可以判断一个无向图中是否存在环，具体步骤如下：

求出图中所有结点的度。
将所有度 <= 1 的结点入队。（独立结点的度为 0）
当队列不空时，弹出队首元素，把与队首元素相邻节点的度减一。如果相邻节点的度变为一，则将相邻结点入队。
循环结束时判断已经访问的结点数是否等于 n。等于 n 说明全部结点都被访问过，无环；反之，则有环。
2.有向图
使用拓扑排序判断无向图和有向图中是否存在环的区别在于：

在判断无向图中是否存在环时，是将所有度 <= 1 的结点入队；
在判断有向图中是否存在环时，是将所有入度 = 0 的结点入队。

```python
from collections import deque, defaultdict
def topological_sort(vertices, edges):
    # 计算所有顶点的入度
    in_degree = {v: 0 for v in vertices}
    graph = defaultdict(list)
	
    # u->v
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1  # v的入度+1

    # 将所有入度为0的顶点加入队列
    queue = deque([v for v in vertices if in_degree[v] == 0])
    sorted_order = []

    while queue:
        u = queue.popleft()
        sorted_order.append(u)

        # 对于每一个相邻顶点，减少其入度
        for v in graph[u]:
            in_degree[v] -= 1
            # 如果入度减为0，则加入队列
            if in_degree[v] == 0:
                queue.append(v)

    if len(sorted_order) != len(vertices):
        return None  # 存在环，无法进行拓扑排序
    return sorted_order


# 示例使用
vertices = ['A', 'B', 'C', 'D', 'E', 'F']
edges = [('A', 'D'), ('F', 'B'), ('B', 'D'), ('F', 'A'), ('D', 'C')]
result = topological_sort(vertices, edges)
if result:
    print("拓扑排序结果:", result)
else:
    print("图中有环，无法进行拓扑排序")
```

```python
#输出全部拓扑序列
from collections import defaultdict, deque
def all_topological_sorts(graph):
    # 计算所有节点的入度
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # 找到所有入度为0的节点
    start_nodes = deque([k for k in in_degree if in_degree[k] == 0])
    result = []
    sort_helper(graph, in_degree, start_nodes, [], result)
    return result

def sort_helper(graph, in_degree, start_nodes, path, result):
    if not start_nodes:
        if len(path) == len(in_degree):
            result.append(list(path))
        return

    for node in list(start_nodes):
        # 将当前节点添加到路径中
        path.append(node)
        start_nodes.remove(node)
        # 减少当前节点指向的所有节点的入度
        for m in graph[node]:
            in_degree[m] -= 1
            if in_degree[m] == 0:
                start_nodes.append(m)
        # 递归调用
        sort_helper(graph, in_degree, start_nodes, path, result)
        # 回溯
        path.pop()
        start_nodes.append(node)
        for m in graph[node]:
            in_degree[m] += 1
            if in_degree[m] == 1:
                start_nodes.remove(m)
# 示例图
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}

# 打印所有拓扑排序
sorts = all_topological_sorts(graph)
for sort in sorts:
  if len(sort)!= len(graph.keys()):
    print('not DAG')
    exit()
    print(sort)
```

#### 3.最短路径专题

##### （1）Dijkstra 算法（从某一点到其他所有点的最短路径）

**Dijkstra算法**：Dijkstra算法用于解决单源最短路径问题，即从给定源节点到图中所有其他节点的最短路径。算法的基本思想是通过不断扩展离源节点最近的节点来逐步确定最短路径。具体步骤如下：

- 初始化一个距离数组，用于记录源节点到所有其他节点的最短距离。初始时，==源节点的距离为0，其他节点的距离为无穷大==。
- 选择一个未访问的节点中距离最小的节点作为当前节点。
- 更新当前节点的邻居节点的距离，如果通过当前节点到达邻居节点的路径比已知最短路径更短，则==更新最短路径==。
- 标记当前节点为已访问。
- 重复上述步骤，直到所有节点都被访问或者所有节点的最短路径都被确定。

Dijkstra算法的时间复杂度为O(V^2)，其中V是图中的节点数。当使用优先队列（如最小堆）来选择距离最小的节点时，可以将时间复杂度优化到O((V+E)logV)，其中E是图中的边数。

```python
# 03424: Candies
# http://cs101.openjudge.cn/practice/03424/
import heapq
def dijkstra(N, G, start):
    INF = float('inf')
    dist = [INF] * (N + 1)  # 存储源点到各个节点的最短距离
    dist[start] = 0  # 源点到自身的距离为0
    pq = [(0, start)]  # 使用优先队列，存储节点的最短距离
    while pq:
        d, node = heapq.heappop(pq)  # 弹出当前最短距离的节点
        if d > dist[node]:  # 如果该节点已经被更新过了，则跳过
            continue
        for neighbor, weight in G[node]:  # 遍历当前节点的所有邻居节点
            new_dist = dist[node] + weight  # 计算经当前节点到达邻居节点的距离
            if new_dist < dist[neighbor]:  # 如果新距离小于已知最短距离，则更新最短距离
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))  # 将邻居节点加入优先队列
    return dist



N, M = map(int, input().split())
G = [[] for _ in range(N + 1)]  # 图的邻接表表示
for _ in range(M):
    s, e, w = map(int, input().split())
    G[s].append((e, w))


start_node = 1  # 源点
shortest_distances = dijkstra(N, G, start_node)  # 计算源点到各个节点的最短距离
print(shortest_distances[-1])  # 输出结果
```

矩阵形式——走山路

```python
#heap局部最优
from heapq import heappop, heappush

def bfs(x1, y1):
    q = [(0, x1, y1)]
    v = set()
    while q:
        t, x, y = heappop(q)
        v.add((x, y))
        if x == x2 and y == y2:
            return t
        for dx, dy in dir:
            nx, ny = x+dx, y+dy
            if 0 <= nx < m and 0 <= ny < n and ma[nx][ny] != '#' and (nx, ny) not in v:
                nt = t+abs(int(ma[nx][ny])-int(ma[x][y]))
                heappush(q, (nt, nx, ny))
    return 'NO'


m, n, p = map(int, input().split())
ma = [list(input().split()) for _ in range(m)]
dir = [(1, 0), (-1, 0), (0, 1), (0, -1)]
for _ in range(p):
    x1, y1, x2, y2 = map(int, input().split())
    if ma[x1][y1] == '#' or ma[x2][y2] == '#':
        print('NO')
        continue
    print(bfs(x1, y1))

```

```python
#通过堆排序的Dijkstra算法（某点到指定顶点的最小路径）
#兔子与樱花
import heapq
def dijkstra(graph,start,end):
    if start == end: 
        return []
    dist = {i:(99999999,[]) for i in graph} #dist字典用于储存顶点名，与起点到顶点的最短距离（权值）以及[]中的路径
    dist[start] = (0,[start])
    pos = [] 
    heapq.heappush(pos,(0,start,[]))
    while pos:
        dist1,current,path = heapq.heappop(pos) #dist1表示起点到上一个相邻节点的距离
        for (next,dist2) in graph[current].items():
            if dist2+dist1 < dist[next][0]:
                dist[next] = (dist2+dist1,path+[next])
                heapq.heappush(pos,(dist1+dist2,next,path+[next]))
    return dist[end][1]

P = int(input())
graph = {input():{} for _ in range(P)} #构建出的图形式为双重字典，每个key代表图中的node，value为键节点所对应所有节点的字典，字典中为对应的路径与权值。
for _ in range(int(input())):
    place1,place2,dist = input().split()
    graph[place1][place2] = graph[place2][place1] = int(dist)

for _ in range(int(input())):
    start,end = input().split()
    path = dijkstra(graph,start,end)
    s = start
    current = start
    for i in path:
        s += f'->({graph[current][i]})->{i}'
        current = i
    print(s)
```

##### （2）Bellman-Ford算法

**Bellman-Ford算法**：Bellman-Ford算法用于解决单源最短路径问题，与Dijkstra算法不同，它可以处理带有负权边的图。算法的基本思想是通过松弛操作逐步更新节点的最短路径估计值，直到收敛到最终结果。具体步骤如下：

- 初始化一个距离数组，用于记录源节点到所有其他节点的最短距离。初始时，源节点的距离为0，其他节点的距离为无穷大。
- 进行V-1次循环（V是图中的节点数），每次循环对所有边进行松弛操作。如果从节点u到节点v的路径经过节点u的距离加上边(u, v)的权重比当前已知的从源节点到节点v的最短路径更短，则更新最短路径。
- 检查是否存在负权回路。如果在V-1次循环后，仍然可以通过松弛操作更新最短路径，则说明存在负权回路，因此无法确定最短路径。

Bellman-Ford算法的时间复杂度为O(V*E)，其中V是图中的节点数，E是图中的边数。

```python
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def bellman_ford(self, src):
        # 初始化距离数组，表示从源点到各个顶点的最短距离
        dist = [float('inf')] * self.V
        dist[src] = 0

        # 迭代 V-1 次，每次更新所有边
        for _ in range(self.V - 1):
            for u, v, w in self.graph:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        # 检测负权环
        for u, v, w in self.graph:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                return "Graph contains negative weight cycle"

        return dist

# 测试代码
g = Graph(5)
g.add_edge(0, 1, -1)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 3)
g.add_edge(1, 3, 2)
g.add_edge(1, 4, 2)
g.add_edge(3, 2, 5)
g.add_edge(3, 1, 1)
g.add_edge(4, 3, -3)

src = 0
distances = g.bellman_ford(src)
print("最短路径距离：")
for i in range(len(distances)):
    print(f"从源点 {src} 到顶点 {i} 的最短距离为：{distances[i]}")

```

##### （3）多源最短路径Floyd-Warshall算法

求解所有顶点之间的最短路径可以使用**Floyd-Warshall算法**，它是一种多源最短路径算法。Floyd-Warshall算法可以**在有向图或无向图中找到任意两个顶点之间的最短路径。**

算法的基本思想是通过一个二维数组来存储任意两个顶点之间的最短距离。初始时，这个数组包含图中各个顶点之间的直接边的权重，对于不直接相连的顶点，权重为无穷大。然后，通过迭代更新这个数组，逐步求得所有顶点之间的最短路径。

具体步骤如下：

1. 初始化一个二维数组`dist`，用于存储任意两个顶点之间的最短距离。初始时，`dist[i][j]`表示顶点i到顶点j的直接边的权重，如果i和j不直接相连，则权重为无穷大。

2. 对于每个顶点k，在更新`dist`数组时，考虑顶点k作为中间节点的情况。遍历所有的顶点对(i, j)，如果通过顶点k可以使得从顶点i到顶点j的路径变短，则更新`dist[i][j]`为更小的值。

   `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`

3. 重复进行上述步骤，对于每个顶点作为中间节点，进行迭代更新`dist`数组。最终，`dist`数组中存储的就是所有顶点之间的最短路径。

Floyd-Warshall算法的时间复杂度为O(V^3)，其中V是图中的顶点数。它适用于解决稠密图（边数较多）的最短路径问题，并且可以处理负权边和负权回路。

以下是一个使用Floyd-Warshall算法求解所有顶点之间最短路径的示例代码：

```python
def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif j in graph[i]:
                dist[i][j] = graph[i][j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist
```

在上述代码中，`graph`是一个字典，用于表示图的邻接关系。它的键表示起始顶点，值表示一个字典，其中键表示终点顶点，值表示对应边的权重。

你可以将你的图表示为一个邻接矩阵或邻接表，并将其作为参数传递给`floyd_warshall`函数。函数将返回一个二维数组，其中`dist[i][j]`表示从顶点i到顶点j的最短路径长度。

##### （4）最小生成树

**1.krustal算法**

Kruskal算法是一种用于解决最小生成树（Minimum Spanning Tree，简称MST）问题的贪心算法。给定一个连通的带权无向图，Kruskal算法可以找到一个包含所有顶点的最小生成树，即包含所有顶点且边权重之和最小的树。

1. 将图中的所有边按照权重从小到大进行排序。（队列）

2. 初始化一个空的边集，用于存储最小生成树的边。

3. 重复以下步骤，直到边集中的边数等于顶点数减一或者所有边都已经考虑完毕：（用并查集判断新加入的边是否合法）

   选择排序后的边集中权重最小的边。

   如果选择的边不会导致形成环路（即加入该边后，两个顶点不在同一个连通分量中），则将该边加入最小生成树的边集中。

4. 返回最小生成树的边集作为结果。

Kruskal算法的核心思想是通过不断选择权重最小的边，并判断是否会形成环路来构建最小生成树。算法开始时，每个顶点都是一个独立的连通分量，随着边的不断加入，不同的连通分量逐渐合并为一个连通分量，直到最终形成最小生成树。

实现Kruskal算法时，一种常用的数据结构是并查集（Disjoint Set）。并查集可以高效地判断两个顶点是否在同一个连通分量中，并将不同的连通分量合并。

```python
#class DisjointSet:
def kruskal(graph):
    num_vertices = len(graph)
    edges = []
    # 构建边集
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))
    # 按照权重排序
    edges.sort(key=lambda x: x[2])
    # 初始化并查集
    disjoint_set = DisjointSet(num_vertices)
    # 构建最小生成树的边集
    minimum_spanning_tree = []
    for edge in edges:
        u, v, weight = edge
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            minimum_spanning_tree.append((u, v, weight))

    return minimum_spanning_tree
```

**2.Prim算法**

普里姆算法（Prim’s algorithm）是一种用于在加权无向图中找到最小生成树（MST）的贪心算法。最小生成树是图中包含所有顶点的边的子集，且这些边的总权重尽可能小。

1. 从图中任意选择一个顶点作为最小生成树的起始点。
2. 维护两个顶点集合：已经包含在MST中的顶点集合和尚未包含在MST中的顶点集合。
3. 在每一步中，考虑连接这两个集合的所有边，并从中选择权重最小的边。
4. 将这条边的另一个端点移动到包含MST的顶点集合中。
5. 重复步骤3，直到所有顶点都包含在MST中。

普里姆算法的关键点在于，它始终保持一个顶点集合已经包含在MST中，然后逐步扩展这个集合，直到包含图中的所有顶点。在每一步中，算法都会选择连接已有MST和新顶点的最小权重边，从而保证了生成树的总权重最小。

```python
# 01258: Agri-Net
# http://cs101.openjudge.cn/practice/01258/
from heapq import heappop, heappush, heapify

def prim(graph, start_node):
    mst = set()
    visited = set([start_node])
    edges = [
        (cost, start_node, to)#边长、起点、终点
        for to, cost in graph[start_node].items()
    ]
    heapify(edges)

    while edges:
        cost, frm, to = heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.add((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in visited:
                    heappush(edges, (cost2, to, to_next))

    return mst
#mst中各个元素的第三项是每条边的长度
```

#### 4.连通性、成环性

##### Kosaraju's算法（有向图连通域）

用于查找有向图中强连通分量（任意两个节点都可到达的一组节点）

```python
def dfs1(graph, node, visited, stack):# 第一个深度优先搜索函数，用于遍历图并将节点按完成时间压入栈中
	visited[node] = True # 标记当前节点为已访问
	for neighbor in graph[node]: # 遍历当前节点的邻居节点
        if not visited[neighbor]: # 如果邻居节点未被访问过
			dfs1(graph, neighbor, visited, stack) # 递归调用深度优先搜索函数
	stack.append(node) # 将当前节点压入栈中，记录完成时间

def dfs2(graph, node, visited, component):# 第二个深度优先搜索函数，用于在转置后的图上查找强连通分量
	visited[node] = True # 标记当前节点为已访问
	component.append(node) # 将当前节点添加到当前强连通分量中
	for neighbor in graph[node]: # 遍历当前节点的邻居节点
		if not visited[neighbor]: # 如果邻居节点未被访问过
			dfs2(graph, neighbor, visited, component) # 递归调用深度优先搜索函数
def kosaraju(graph):# Kosaraju's 算法函数
	# Step 1: 执行第一次深度优先搜索以获取完成时间
	stack = [] # 用于存储节点的栈
	visited = [False] * len(graph) # 记录节点是否被访问过的列表
	for node in range(len(graph)): # 遍历所有节点
		if not visited[node]: # 如果节点未被访问过
			dfs1(graph, node, visited, stack) # 调用第一个深度优先搜索函数
	
    # Step 2: 转置图
	transposed_graph = [[] for _ in range(len(graph))] # 创建一个转置后的图
		for node in range(len(graph)): # 遍历原图中的所有节点
			for neighbor in graph[node]: # 遍历每个节点的邻居节点
				transposed_graph[neighbor].append(node) # 将原图中的边反向添加到转置图中

    # Step 3: 在转置后的图上执行第二次深度优先搜索以找到强连通分量
	visited = [False] * len(graph) # 重新初始化节点是否被访问过的列表
	sccs = [] # 存储强连通分量的列表
	while stack: # 当栈不为空时循环
		node = stack.pop() # 从栈中弹出一个节点
		if not visited[node]: # 如果节点未被访问过
			scc = [] # 创建一个新的强连通分量列表
			dfs2(transposed_graph, node, visited, scc) # 在转置图上执行深度优先搜索
			sccs.append(scc) # 将找到的强连通分量添加到结果列表中
	return sccs # 返回所有强连通分量的列表
```

##### 另：判断无向图是否连通有无回路

思路：比较简单的并查集方法，直接将祖先中大的应该指向小的，如果在过程中有遇到某边的两个连接点是指向同一祖先的说明出现了连通情况loop=yes，最后统计根节点的数量如果只有1个根节点说明只有一个回路connected=yes

代码

```python
#赵语涵2300012254
n,m = map(int,input().split())
parent = [x for x in range(n)]
def find(x):
    if parent[x] != x:
        return find(parent[x])
    return parent[x]
loop = 'no'
for _ in range(m):
    a,b = map(int,input().split())
    x,y = find(a),find(b)
    if x == y:
        loop = 'yes'
    else:
        if x < y:
            parent[y] = x
        else:
            parent[x] = y
ancient = set(find(x) for x in range(n))
if len(ancient)==1:
    print('connected:yes')
else:
    print('connected:no')
print(f'loop:{loop}')
```





### 三、并查集

并查集是一种用于处理不相交集合的数据结构，通常用于解决一些与集合操作有关的问题，例如连通性、图的最小生成树、社交网络中的关系等。

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n#表示集合大小，优化并查集效率

    def find(self, x):#查找根节点
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):#合并两个集合
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
    # 计算并查集中不同集合的数量
    def count(self):
        # 通过查找每个元素的根节点，并去重，得到不同集合的数量
        return len(set(self.find(i) for i in range(len(self.root))))
      
'应用'
		#检验两个结点是否联通
		def joined(self,a,b):
      return self.find(a)==self.find(b)
    #找出图中成环时最后加入的一条边：加入边时两个点已经联通，则成环了
    
```

### 四、树

#### 概念

**层级 Level**：
从根节点开始到达一个节点的路径，所包含的==边的数量==，称为这个节点的层级。
如图 D 的层级为 2，根节点的层级为 0。

有时候，题目中会给出概念定义，如：

**高度 Height**：树中所有节点的==最大层级==称为树的高度。

**二叉树深度**：从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，==最长路径的节点个数==为树的深度

2.按形态分类（主要是二叉树）

（1）完全二叉树——第n-1层全满，最后一层按顺序排列

（2）满二叉树——二叉树的最下面一层元素全部满就是满二叉树

（3）avl树——平衡因子，左右子树高度差不超过1

​	= =这块一定要弄懂左右旋的概念，理解什么时候左旋什么时候右旋，以及操作的具体过程！

（4）二叉查找树(二叉排序\搜索树)
	= =特点：没有相同键值的节点。

​	= =若左子树不空，那么其所有子孙都比根节点小。

​	= =若右子树不空，那么其所有子孙都比根节点大。

​	= =左右子树也分别为二叉排序树。

（5）哈夫曼树——哈夫曼树是一种针对权值的二叉树。一般为了减少计算机运算速度，将权重大的放在最前面

#### 表示

树的括号嵌套表示转正常表示

```python
class TreeNode:
    def __init__(self, value): #类似字典
        self.value = value
        self.children = []

def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点


def preorder(node):
    output = [node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)

def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)

# 主程序
def main():
    s = input().strip()
    s = ''.join(s.split())  # 去掉所有空白字符
    root = parse_tree(s)  # 解析整棵树
    if root:
        print(preorder(root))  # 输出前序遍历序列
        print(postorder(root))  # 输出后序遍历序列
    else:
        print("input tree string error!")

if __name__ == "__main__":
    main()
```

根据中后序序列建树

```python
"""
定义一个递归函数。在这个递归函数中，我们将后序遍历的最后一个元素作为当前的根节点，然后在中序遍历序列中找到这个根节点的位置，
这个位置将中序遍历序列分为左子树和右子树。
"""
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return None
    # 后序遍历的最后一个元素是当前的根节点
    root_val = postorder.pop()
    root = TreeNode(root_val)
    # 在中序遍历中找到根节点的位置
    root_index = inorder.index(root_val)
    # 构建右子树和左子树
    root.right = buildTree(inorder[root_index + 1:], postorder)
    root.left = buildTree(inorder[:root_index], postorder)
    return root
def preorderTraversal(root):
    result = []
    if root:
        result.append(root.val)
        result.extend(preorderTraversal(root.left))
        result.extend(preorderTraversal(root.right))
    return result
# 读取输入
inorder = input().strip()
postorder = input().strip()
# 构建树
root = buildTree(list(inorder), list(postorder))
# 输出前序遍历序列
print(''.join(preorderTraversal(root)))
```

根据前中序序列建树

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root
def postorder_traversal(root):
    if root is None:
        return ''
    return postorder_traversal(root.left) + postorder_traversal(root.right) + root.value

while True:
    try:
        preorder = input().strip()
        inorder = input().strip()
        root = build_tree(preorder, inorder)
        print(postorder_traversal(root))
    except EOFError:
        break
```

#### 遍历

```python 3.8
@前序中序后序层序遍历
def preorder(root):
    if root is None:
        return []
    result=[]
    result+=[root.val]
    for i in root.children:
        result+=preorder(i)
    return result
def inorder(root):
    result=[]
    if root is not None:
        result+=postorder(root.left)
        result+=[root.val]
        result+=postorder(root.right)
    return result
def postorder(root):
    result=[]
    if root is not None:
        result+=postorder(root.left)
        result+=postorder(root.right)
        result+=[root.val]
    return result
def level(root):
    if root is None:
        return []
    result=[]
    queue=[root]
    while queue:
        node=queue.pop(0)
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result
```

#### 哈弗曼编码

赫夫曼编码是一种前缀码，根据字符出现的概率构造出不等长的二进制编码，使编码后的数据长度最短，同时保证不产生二义性。每个字符的编码由赫夫曼树的路径决定，左孩子为0，右孩子为1。

这段代码首先定义了一个 `Node` 类来表示哈夫曼树的节点。然后，使用最小堆来构建哈夫曼树，每次从堆中取出两个频率最小的节点进行合并，直到堆中只剩下一个节点，即哈夫曼树的根节点。接着，使用递归方法计算哈夫曼树的带权外部路径长度（weighted external path length）。最后，输出计算得到的带权外部路径长度。

霍夫曼树是一种带权路径长度最短的二叉树，也就是说，它的构建过程中会确保所有叶子节点的带权路径长度之和最小。这里的带权路径长度指的是从根节点到每个叶子节点的路径长度与该叶子节点权值的乘积之和。

```python
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight

def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight) #note: 合并后，char 字段默认值是空
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def encode_huffman_tree(root):
    codes = {}

    def traverse(node, code):
        if node.char:
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        if node.char:
            decoded += node.char
            node = root
    return decoded

# 读取输入
n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

#string = input().strip()
#encoded_string = input().strip()

# 构建哈夫曼编码树
huffman_tree = build_huffman_tree(characters)

# 编码和解码
codes = encode_huffman_tree(huffman_tree)

strings = []
while True:
    try:
        line = input()
        if line:
            strings.append(line)
        else:
            break
    except EOFError:
        break

results = []
#print(strings)
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree, string))
    else:
        results.append(huffman_encoding(codes, string))

for result in results:
    print(result)
```

```python
'''选取最小的两个节点合并时，节点比大小的规则是:
1) 权值小的节点算小。权值相同的两个节点，字符集里最小字符小的，算小。
例如 （{'c','k'},12) 和 ({'b','z'},12)，后者小。
2) 合并两个节点时，小的节点必须作为左子节点
3) 连接左子节点的边代表0,连接右子节点的边代表1
然后对输入的串进行编码或解码'''
import heapq
from collections import defaultdict
#创建树节点类
class Node:
    def __init__(self,freq,char=set()):
        self.char=char#节点对应的字符
        self.freq=freq#节点字符出现的频率
        self.left=None
        self.right=None
    def __lt__(self,other):
        #重载小于操作符，用于节点比较
        if self.freq<other.freq:
            return True
        elif self.freq>other.freq:
            return False
        else:
            return min(self.char)<min(other.char)
#创建优先队列并建立哈夫曼树
def create_tree(frequencies):
    #使用频率创建节点，并加入优先队列
    pq=[Node(freq,set(char)) for char,freq in frequencies.items()]
    heapq.heapify(pq)
    while len(pq)>1:
        #循环只到堆中只剩一个节点
        left=heapq.heappop(pq)
        right=heapq.heappop(pq)
        merged=Node(left.freq+right.freq,left.char|right.char)#合并这两个节点
        merged.left=left
        merged.right=right
        heapq.heappush(pq,merged)#将合并后的节点加入堆中
    return pq[0]
#霍夫曼编码
def huffman_encoding(node,prefix='',code={}):
    if node != None:
        if len(node.char)==1:
            code[''.join(node.char)]=prefix
        huffman_encoding(node.left,prefix+'0',code)
        huffman_encoding(node.right,prefix+'1',code)
    return code

def count_char_frequency(text):
    frequency=defaultdict(int)
    for char in text:
        frequency[char]+=1
    return frequency
#编码文本
def encode(text):
    frequencies=count_char_frequency(text)
    root=create_tree(frequencies)
    huffman_code=huffman_encoding(root)
    encoded_text=''.join([huffman_code[char] for char in text])
    return encoded_text,huffman_code
#解码文本
def decode(encoded_text,huffman_code):
    reverse_code={v:k for k,v in huffman_code.items()}
    code=''
    decoded_text=''
    for digit in encoded_text:
        code+=digit
        if code in reverse_code:
            decoded_text+=reverse_code[code]
            code=''
    return decoded_text

n=int(input())
text=''
for i in range(n):
    char,num=input().split()
    num=int(num)
    text+=char*num
encoded_text,huffman_code=encode(text)
output=[]
while True:
    try:
        s=input()
        if s.isdigit():
            output.append(decode(s,huffman_code))
        else:
            encodedresult=''
            for string in s:
                encodedresult+=huffman_code[string]
            output.append(encodedresult)
    except EOFError:
        break
for o in output:
    print(o)
```



#### 二叉堆

二叉堆是一棵**完全二叉树**或近似完全二叉树，每个节点中存储一个元素（或权值）。

堆性质：父节点的权值不小于其子节点的权值（大根堆），或父节点的权值不大于其子节点的权值（小根堆）。

1.每次插入元素从最后插入，然后进行调整堆的操作

2.每次删除元素从堆顶找元素，找到元素后将该元素和最后一个元素换位，然后进行==重排操作==

（这里要求的是删除最小元素，即堆顶元素）

```python
class tree_node:
    def __init__(self):#初始化空堆
        self.heap = []

    def parent(self, i):#返回父节点的索引
        return (i - 1) // 2

    def left_child(self, i):#返回左子节点的索引
        return 2 * i + 1

    def right_child(self, i):#返回右子节点的索引
        return 2 * i + 2

    def swap(self, i, j):#交换堆中两个元素的位置
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def insert(self, item):#插入操作
        self.heap.append(item)
        self.heapify_up(len(self.heap) - 1)

    def delete(self):#删除操作，返回最小元素
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        self.swap(0, len(self.heap) - 1)
        min_value = self.heap.pop()
        self.heapify_down(0)
        return min_value

    def heapify_up(self, i):#向上调整
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def heapify_down(self, i):#向下调整
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)

        if left < len(self.heap) and self.heap[left] < self.heap[min_index]:
            min_index = left

        if right < len(self.heap) and self.heap[right] < self.heap[min_index]:
            min_index = right

        if i != min_index:
            self.swap(i, min_index)
            self.heapify_down(min_index)
n = int(input())
lst = tree_node()
for _ in range(n):
    s = input()
    if s[0] == '1':
        lst.insert(int(s[2:]))
    if s[0] == '2':
        print(lst.delete())
```

1. **`heappush(heap, item)`**:将元素 `item` 推入堆 `heap` 中，保持堆的性质不变。
2. **`heappop(heap)`**:弹出并返回堆 `heap` 中的最小元素，同时保持堆的性质。
3. **`heapify(x)`**:将列表 `x` 转换为一个堆，原地进行，时间复杂度为线性。

```python 3.8
def buildtree(preorder,inorder):
    if not preorder or not inorder:
        return None
    root=Node(preorder[0])
    rootindex=inorder.index(root.val)
    root.left=buildtree(preorder[1:rootindex+1],inorder[:rootindex])
    root.right=buildtree(preorder[rootindex+1:],inorder[rootindex+1:])
    return root
def build(postorder,inorder):
    if not postorder or not inorder:
        return None
    root_val=postorder[-1]
    root=node(root_val)
    mid=inorder.index(root_val)
    root.left=build(postorder[:mid],inorder[:mid])
    root.right=build(postorder[mid:-1],inorder[mid+1:])
    return root
```

#### 二叉搜索树

二叉搜索树性质：小于父节点的键都在左子树中，大于父节点的键则都在右子树中。

根据无序序列建立BST并输出层序遍历

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert_into_bst(root, value):
    if root is None:
        return TreeNode(value)
    if value < root.value:
        root.left = insert_into_bst(root.left, value)
    else:
        root.right = insert_into_bst(root.right, value)
    return root

def build_bst_from_list(values):
    '''根据无序列表构建BST'''
    root = None
    for value in values:
        root = insert_into_bst(root, value)
    return root

l=map(int,input().split())
l=list(dict.fromkeys(l))#去除重复的数字
bst_root=build_bst_from_list(l)
#层次遍历/bfs
output=[]
from collections import deque
q=deque()#借助双端队列实现bfs
q.append(bst_root)
while q:
    size=len(q)
    for i in range(size):
        tmproot=q.popleft()
        output.append(tmproot.value)
        if tmproot.left:
            q.append(tmproot.left)
        if tmproot.right:
            q.append(tmproot.right)
print(' '.join(map(str,output)))
```

#### 平衡二叉树（AVL树）

AVL树是一种自平衡的二叉查找树。它的特点如下：

任何节点的两个子树的高度差的绝对值不超过1，左子树和右子树都是一棵平衡二叉树。

```python
class Node:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
        self.height=1#树的高度
class AVL:
    '''二叉平衡搜索树'''
    def __init__(self):
        self.root=None#self.root表示AVL树根节点
    def insert(self,value):
        '''向avl树中插入值为value的元素'''
        if not self.root:#空树则创建树
            self.root=Node(value)
        else:
            self.root=self._insert(value,self.root)#非空则插入
    def _insert(self,value,node):#value为新插入的点的值
      '_insert()函数的递归当中自底向上地检查祖先节点是否失衡'
        if not node:#空树则建树
            return Node(value)
        elif value<node.value:#应归于左子树
            node.left=self._insert(value,node.left)
        else:#应归于右子树
            node.right=self._insert(value, node.right)
				node.height=1+max(self._get_height(node.left),
                      self._get_height(node.right))
        #二叉树高度的递推式
        
        #调整平衡
        balance = self._get_balance(node)#当前节点的平衡因子
        
        if balance > 1:
            if value < node.left.value:	# 树形是 LL
                return self._rotate_right(node)
            else:	# 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:	# 树形是 RR
                return self._rotate_left(node)
            else:	# 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        return node
    
    def _get_height(self, node):
        '''或许node的树高度'''
        if not node:
            return 0
        return node.height
    def _get_balance(self, node):
        '''获取节点node的平衡因子'''
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_left(self, z):
        '''左旋'''
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        '''右旋'''
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x
    
    def preorder(self):
        '''输出前序遍历'''
        return self._preorder(self.root)

    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str, avl.preorder())))
```

#### 前缀树Trie

前缀树，也称为Trie，是一种用于检索字符串数据集中的键的有序树数据结构。这种数据结构非常适合解决诸如自动补全和拼写检查等问题。在Trie树中，每个节点代表一个字符，从根节点到某一节点的路径对应于数据集中的一个前缀。

您提供的Python代码定义了一个Trie树的基本实现。这里是代码的简要说明：

`TrieNode` 类代表Trie树中的每个节点。每个节点包含：

- `char`：当前节点的字符。
- `is_end`：一个布尔值，表示是否有单词以这个节点结束。
- `children`：一个字典，存储当前节点的子节点。

`Trie` 类代表整个Trie树。它包含：

- `root`：Trie树的根节点，初始化为空字符。
- `insert` 方法：用于将单词添加到Trie树中。
- `dfs` 方法：深度优先搜索，用于找到所有以当前节点为前缀的单词。
- `search` 方法：用于查找以某个字符串为前缀的所有单词。

```python 3.8
class TrieNode:
    def __init__(self, char):
        self.char = char
        self.is_end = False
        self.children = {}
class Trie(object):
    def __init__(self):
        self.root = TrieNode("")
    def insert(self, word):
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node
        node.is_end = True
    def dfs(self, node, pre):
        if node.is_end:
            self.output.append((pre + node.char))
        for child in node.children.values():
            self.dfs(child, pre + node.char)
    def search(self, x):
        node = self.root
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        self.output = []
        self.dfs(node, x[:-1])
        #print(x)
        #print(self.output)
        return self.output
```

#### 解析树

这个解析树使用二叉树来表示表达式中的操作符和操作数，其中每个节点可以代表一个操作数（如数字）或一个操作符（如 `+`, `-`, `*`, `/`）。代码中的 `buildParseTree` 函数将一个完全括号化的数学表达式转换成一个解析树。

`Stack` 类是一个标准的栈实现，用于在构建解析树时存储父节点的引用。

`BinaryTree` 类代表解析树中的每个节点，可以插入左右子节点，并可以进行前序、中序和后序遍历。

`buildParseTree` 函数读取一个表达式字符串，创建一个解析树。

`evaluate` 函数计算解析树的数值结果。

`postordereval` 函数是后序遍历求值的实现。

`printexp` 函数将解析树转换回完全括号化的表达式。

这段代码的执行结果是将表达式 `"( ( 7 + 3 ) * ( 5 - 2 ) )"` 转换成解析树，并计算其结果，输出为 `30`。同时，它还能够以前序、中序和后序遍历的方式打印出表达式的不同形式。

```python 3.8
# 定义一个栈类
class Stack(object):
    def __init__(self):
        self.items = []  # 初始化空列表作为栈的存储结构
        self.stack_size = 0  # 栈的大小
    # 判断栈是否为空
    def isEmpty(self):
        return self.stack_size == 0
    # 向栈中添加元素
    def push(self, new_item):
        self.items.append(new_item)
        self.stack_size += 1
    # 从栈中弹出元素
    def pop(self):
        self.stack_size -= 1
        return self.items.pop()
    # 查看栈顶元素
    def peek(self):
        return self.items[self.stack_size - 1]
    # 获取栈的大小
    def size(self):
        return self.stack_size
# 定义一个二叉树类，用于构建解析树
class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj  # 根节点的值
        self.leftChild = None  # 左子节点
        self.rightChild = None  # 右子节点
    # 插入左子节点
    def insertLeft(self, newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:  # 如果已经有左子节点，则将新节点插入为当前左子节点的左子节点
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t
    # 插入右子节点
    def insertRight(self, newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:  # 如果已经有右子节点，则将新节点插入为当前右子节点的右子节点
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t
    # 获取右子节点
    def getRightChild(self):
        return self.rightChild
    # 获取左子节点
    def getLeftChild(self):
        return self.leftChild
    # 设置根节点的值
    def setRootVal(self, obj):
        self.key = obj
    # 获取根节点的值
    def getRootVal(self):
        return self.key
    # 遍历解析树
    def traversal(self, method="preorder"):
        if method == "preorder":
            print(self.key, end=" ")
        if self.leftChild != None:
            self.leftChild.traversal(method)
        if method == "inorder":
            print(self.key, end=" ")
        if self.rightChild != None:
            self.rightChild.traversal(method)
        if method == "postorder":
            print(self.key, end=" ")
# 构建解析树的函数
def buildParseTree(fpexp):
    fplist = fpexp.split()  # 将表达式分割成列表
    pStack = Stack()  # 创建一个栈
    eTree = BinaryTree('')  # 创建一个空的二叉树
    pStack.push(eTree)  # 将二叉树压入栈
    currentTree = eTree  # 设置当前树为二叉树
    for i in fplist:
        if i == '(':
            currentTree.insertLeft('')  # 遇到左括号，插入一个新的左子节点
            pStack.push(currentTree)  # 将当前节点压入栈
            currentTree = currentTree.getLeftChild()  # 移动到左子节点
        elif i not in '+-*/)':
            currentTree.setRootVal(int(i))  # 如果是数字，设置当前节点的值
            parent = pStack.pop()  # 弹出父节点
            currentTree = parent  # 移动到父节点
        elif i in '+-*/':
            currentTree.setRootVal(i)  # 如果是操作符，设置当前节点的值
            currentTree.insertRight('')  # 插入一个新的右子节点
            pStack.push(currentTree)  # 将当前节点压入栈
            currentTree = currentTree.getRightChild()  # 移动到右子节点
        elif i == ')':
            currentTree = pStack.pop()  # 遇到右括号，弹出当前节点
        else:
            raise ValueError("Unknown Operator: " + i)  # 遇到未知操作符，抛出异常
    return eTree  # 返回构建好的解析树
# 示例表达式
exp = "( ( 7 + 3 ) * ( 5 - 2 ) )"
pt = buildParseTree(exp)  # 构建解析树
for mode in ["preorder", "postorder", "inorder"]:  # 遍历解析树
    pt.traversal(mode)
    print()
# 计算解析树的值的函数
def evaluate(parseTree):
    opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}  # 定义操作符和对应的函数
    leftC = parseTree.getLeftChild()  # 获取左子节点
    rightC = parseTree.getRightChild()  # 获取右子节点
    if leftC and rightC:  # 如果左右子节点都存在
        fn = opers[parseTree.getRootVal()]  # 获取操作符对应的函数
        return fn(evaluate(leftC),evaluate(rightC))  # 递归计算左右子节点的值，并应用操作符
    else:
        return parseTree.getRootVal()  # 如果是叶子节点，返回节点的值

print(evaluate(pt))  # 打印解析树的计算结果
# 后序遍历求值的函数
def postordereval(tree):
    opers = {'+':operator.add, '-':operator.sub,
             '*':operator.mul, '/':operator.truediv}  # 定义操作符和对应的函数
    res1 = None
    res2 = None
    if tree:  # 如果树不为空
        res1 = postordereval(tree.getLeftChild())  # 递归计算左子节点的值
        res2 = postordereval(tree.getRightChild())  # 递归计算右子节点的值
        if res1 and res2:  # 如果左右子节点的值都存在
            return operstree.getRootVal()  # 应用操作符
        else:
            return tree.getRootVal()  # 如果是叶子节点，返回节点的值

print(postordereval(pt))  # 打印后序遍历求值的结果

# 中序遍历还原完全括号表达式的函数
def printexp(tree):
    sVal = ""  # 初始化空字符串
    if tree:  # 如果树不为空
        sVal = '(' + printexp(tree.getLeftChild())  # 递归添加左子节点的表达式
        sVal = sVal + str(tree.getRootVal())  # 添加根节点的值
        sVal = sVal + printexp(tree.getRightChild()) + ')'  # 递归添加右子节点的表达式
    return sVal  # 返回完整的表达式

print(printexp(pt))  # 打印完全括号化的表达式
```

### 五、其他

#### 1.DFS

```py
#寻宝 dfs版
def dfs(x,y):
    dx=[1,0,-1,0]
    dy=[0,1,0,-1]
    global cnt
    global cnt_min
    global treasure
    for i in range(4):
        if treasure[x+dx[i]][y+dy[i]]!=2:
            treasure[x][y]=2
            cnt+=1
            if treasure[x+dx[i]][y+dy[i]]==1:
                cnt_min=min(cnt,cnt_min)
                treasure[x][y]=0  #注意计数器和图都要恢复回溯
                cnt-=1
                break
            else:
                dfs(x+dx[i],y+dy[i])
                treasure[x][y]=0
                cnt-=1
    return

cnt_min=99999
cnt=0
n,m=map(int,input().split())
treasure=[[2]*(m+2)]
for i in range(n):
    treasure.append([2]+[int(j) for j in input().split()]+[2])
treasure.append([2]*(m+2))
if treasure[1][1]==1:
    print(0)
    exit()
dfs(1,1)
if cnt_min==99999:
    print("NO")
else:
    print(cnt_min)
```

```py
#八皇后问题
def dfs(cur):
    global solution
    for i0 in range(8):
        flag=True
        for j0 in range(len(solution)): #用列表solution直接储存某一次的结果，因为八皇后可以直接根据前面的路径判断后面的位置是否可访问，不需要在图上修改
            if cur+i0==j0+solution[j0] or cur-i0==j0-solution[j0] or i0 in solution:
                flag=False
        if flag:
            solution+=[i0]
            if cur==7:
                su=0
                for i1 in range(8):
                    su+=(solution[i1]+1)*10**(7-i1)
                ans.append(su)
                solution.pop(-1)
            else:
                dfs(cur+1)
                solution.pop(-1)
        else:
            continue

ans=[]
solution=[]
dfs(0)
n=int(input())
for i in range(n):
    num=int(input())
    print(ans[num-1])
```

```py
#组合乘积
def dfs(T,l): #dfs解决非图问题
    for i in range(len(l)):
        if l[i]!=0:
            if T%l[i]==0 and l[i]>0:
                T=T//l[i]
                temp=l[i]
                l[i]=-1  #防止路径重复
                if T==1:
                    print("YES")
                    exit()
                else:
                    dfs(T,l)
                    l[i]=temp  #零时储存前一个值便于回溯
                    T=T*temp

t=int(input())
num=[int(i) for i in input().split()]
if t==0:
    if 0 in num:
        print("YES")
    else:
        print("NO")
else:
    dfs(t,num)
    print("NO")
```

```python
#迷宫最大权值和
dx = [-1, 0, 1, 0]
dy = [ 0, 1, 0, -1]
maxValue = -9999
def dfs(maze, x, y, nowValue):
    global maxValue
    if x==n and y==m:
        if nowValue > maxValue:
            maxValue = nowValue
            return
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if maze[nx][ny] == 0:
            maze[nx][ny] = -1
            tmp = w[x][y]
            w[x][y] = -9999
            nextValue = nowValue + w[nx][ny]
            dfs(maze, nx, ny, nextValue)
            maze[nx][ny] = 0
            w[x][y] = tmp

n, m = map(int, input().split())
maze = []
maze.append( [-1 for x in range(m+2)] )
for _ in range(n):
    maze.append([-1] + [int(_) for _ in input().split()] + [-1])
maze.append([-1 for x in range(m + 2)])
w = []
w.append([-9999 for x in range(m + 2)])
for _ in range(n):
    w.append([-9999] + [int(_) for _ in input().split()] + [-9999])
w.append([-9999 for x in range(m + 2)])
dfs(maze, 1, 1, w[1][1])
print(maxValue)
```

#### 2.BFS

```py
#寻宝 bfs版，bfs寻找最短路径长度
from collections import deque
def valid(x,y):
    if 0<=x<m and 0<=y<n:
        if treasure[x][y]!=2 and not inque[x][y]:
            return True
        else:
            return False
    else:
        return False

def bfs():
    q=deque()
    q.append((0,0,0)) #将目前格子的步数和位置一同记录
    while q:
        t=q.popleft()
        for i in range(4):
            x,y,cnt=t[0]+dx[i],t[1]+dy[i],t[2]
            if valid(x,y):
                inque[x][y]=True
                q.append((x,y,cnt+1)) #在前一格的基础上cnt+1
                if treasure[x][y]==1:
                    print(cnt+1)
                    return
    print("NO")

m,n=map(int,input().split())
treasure=[]
for i in range(m):
    x=[int(_) for _ in input().split()]
    treasure.append(x)
inque=[[False]*n for i in range(m)]
dx=[0,1,0,-1]
dy=[1,0,-1,0]
if treasure[0][0]==1:
    print(0)
    exit()
if treasure[0][0]==2:
    print("NO")
    exit()
bfs()
```

```py
#最大连通域面积-bfs模板-面积计算
from collections import deque
dx=[1,-1,0,0,1,1,-1,-1]
dy=[0,0,1,-1,1,-1,-1,1]
max_total=0
def valid(x,y):
    if 0<=x<n and 0<=y<m:
        if matrix[x][y]=="W" and not inq[x][y]:
            return True
        else:
            return False
    else:
        return False
def bfs(x,y):
    global max_total  #多组数据时全局变量一定不要忘记归零！
    q=deque()
    q.append((x,y))
    inq[x][y]=True
    cnt=1
    while q:
        bounce=q.popleft()
        for r in range(8):
            nx=bounce[0]+dx[r]
            ny=bounce[1]+dy[r]
            if valid(nx,ny):
                q.append((nx,ny))
                inq[nx][ny]=True
                cnt+=1
    max_total=max(max_total,cnt)

t=int(input())
for i in range(t):
    n,m=map(int,input().split())
    matrix=[]
    for i0 in range(n):
        matrix.append(input())
    inq=[]
    for i1 in range(n):
        inq.append([False]*m)
    for k1 in range(n):
        for k2 in range(m):
            if valid(k1,k2):
                bfs(k1,k2)
    print(max_total)
    max_total=0
```

```py
#迷宫最短路径
#如何在bfs中记录路径
from queue import Queue
MAXN = 100
MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
def canVisit(x, y):
    return x >= 0 and x < n and y >= 0 and y < m and maze[x][y] == 0 and not inQueue[x][y]
def BFS(x, y):
    q = Queue()
    q.put((x, y))
    inQueue[x][y] = True
    while not q.empty():
        front = q.get()
        if front[0] == n - 1 and front[1] == m - 1:
            return
        for i in range(MAXD):
            nextX = front[0] + dx[i]
            nextY = front[1] + dy[i]
            if canVisit(nextX, nextY):
                pre[nextX][nextY] = (front[0], front[1]) #pre中每个点都记录前一个点的位置
                inQueue[nextX][nextY] = True
                q.put((nextX, nextY))
def printPath(p):
    prePosition = pre[p[0]][p[1]]
    if prePosition == (-1, -1):
        print(p[0] + 1, p[1] + 1)
        return
    printPath(prePosition)  #类似回溯，推到起始点时开始从头输出
    print(p[0] + 1, p[1] + 1)

n, m = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)
inQueue = [[False] * m for _ in range(n)]
pre = [[(-1, -1)] * m for _ in range(n)]
BFS(0, 0)
printPath((n - 1, m - 1))
```

#### 

#### 4.dp

##### 1.背包问题

0-1 背包：有n种物品，每种物品只有一个。每个物品有自己的重量和价值。有一个给定容量的背包，问这个背包最多能装的最大价值是多少。

Solution 1 : 二维数组

```python 3.8
# n, v分别代表物品数量，背包容积
n, v = map(int, input().split())
# w为物品价值，c为物品体积（花费）
w, cost = [0], [0]
for i in range(n):
    cur_c, cur_w = map(int, input().split())
    w.append(cur_w)
    cost.append(cur_c)

#该初始化代表背包不一定要装满
dp = [[0 for j in range(v+1)] for i in range(n+1)]

for i in range(1, n+1):
    for j in range(1, v+1):    #可优化成 for j in range(cost[i], v+1): 
        if j < cost[i]:
            dp[i][j] = dp[i-1][j]
        else:
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-cost[i]]+w[i])
print(dp[n][v])
```

Solution 2 : 滚动数组

```python 3.8
# n, v分别代表物品数量，背包容积
n, v = map(int, input().split())
# w为物品价值，c为物品体积（花费）
w, cost = [0], [0]
for i in range(n):
    cur_c, cur_w = map(int, input().split())
    w.append(cur_w)
    cost.append(cur_c)

#该初始化代表背包不一定要装满
dp = [0 for j in range(v+1)]

for i in range(1, n+1):
    #注意：第二层循环要逆序循环
    for j in range(v, 0, -1):       #可优化成 for j in range(v, cost[i]-1, -1): 
        if j >= cost[i]:#否则j<cost[i],dp[i][j]=dp[i-1][j],也就是dp[j]无需更新
            dp[j] = max(dp[j], dp[j-cost[i]]+w[i])
		
print(dp[v])
```

##### 2.完全背包问题

有N种物品和一个容量是V的背包，每种物品都有无限件可用。第i种物品的体积是vi，价值是wi。求解将哪些物品装入背包，可使这些物品的总体积不超过背包容量，且总价值最大，求出最大总价值。

Solution ：滚动数组

```python 3.8
# n, v分别代表物品数量，背包容积
n, v = map(int, input().split())
# w为物品价值，c为物品体积（花费）
w, cost = [0], [0]
for i in range(n):
    cur_c, cur_w = map(int, input().split())
    w.append(cur_w)
    cost.append(cur_c)

#该初始化代表背包不一定要装满
dp = [0 for j in range(v+1)]

for i in range(1, n+1):
    #只需要将0-1背包一维DP解法中的二层循环改为顺序循环
    for j in range(1, v+1):
        if j >= cost[i]:
            dp[j] = max(dp[j], dp[j-cost[i]]+w[i])

print(dp[v])
```

==为什么顺序遍历就能解决问题？对于当前物品 i ,要么不拿为dp[j]，要么拿为dp[j-cost[i]]+w[i]，而dp[j-cost[i]]代表之前的状态中，也包含拿过物品 i 的状态，这样就包含了多次拿取物品 i 的情况。==

优化：若两件物品A，B满足A的体积大于B并且A的价值不大于B，那么可以直接排除使用A的可能。优化复杂度在$O(n^2)$

##### 3.多重背包问题

有N种物品和一个容量是V的背包。第i种物品最多有si件，每件体积是vi，价值是wi。将哪些物品装入背包，可使物品总体积和不超过背包容量，且总价值和最大。

Solution 1： 0-1背包的变式：==将每件物品的件数 Mi 作为独立的物品==

```python3.8
# n, v分别代表物品数量，背包容积
n, v = map(int, input().split())
# w为物品价值，c为物品体积（花费）
w, cost, s = [0], [0], [0]
for i in range(n):
    cur_c, cur_w,cur_s= map(int, input().split())
    w += [cur_w]*cur_s
    cost += [cur_c]*cur_s

n = len(w)-1

#该初始化代表背包不一定要装满
dp = [0 for j in range(v+1)]

for i in range(1, n+1):
    for j in range(v, cost[i]-1, -1):
        if j >= cost[i]:
            dp[j] = max(dp[j], dp[j-cost[i]]+w[i])

print(dp[v])
```

Solution 2 ：优化的转化到0-1背包问题

```python 3.8
class Solution:
    # 0-1背包问题的写法
    def max_value(self, n, m, v, w):
        dp = [0] * (m + 1)
        for i in range(1, n + 1):
            for j in range(m, v[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - v[i]] + w[i])
        return dp[-1]


if __name__ == '__main__':
    import sys

    n, m = map(int, input().split())
    lines = sys.stdin.readlines()
    v, w = [0], [0]
    n = 0
    for line in lines:
        line = list(map(int, line.split()))
        k = 1
        while k <= line[2]:  # 假设line[2]=13,k取1,2,4之后，line[2] = 6 < k = 8 退出循环
            v.append(k * line[0])
            w.append(k * line[1])
            line[2] -= k
            k *= 2
            n += 1  # 物品总数加1
        if line[2]:
            v.append(line[2] * line[0])
            w.append(line[2] * line[1])
            n += 1
    print(Solution().max_value(n, m, v, w))
```

**最大上升子序列**

```python
input()
b = [int(x) for x in input().split()]

n = len(b)
dp = [0]*n

for i in range(n):
    dp[i] = b[i]
    for j in range(i):
        if b[j]<b[i]:
            dp[i] = max(dp[j]+b[i], dp[i])
    
print(max(dp))
```

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



### 六、一些技巧&算法

## 堆

堆是一种特殊的[树形数据结构](https://so.csdn.net/so/search?q=树形数据结构&spm=1001.2101.3001.7020)，其中每个节点的值都小于或等于（最小堆）或大于或等于（最大堆）其子节点的值。堆分为最小堆和最大堆两种类型，其中：

- 最小堆： 父节点的值小于或等于其子节点的值。
- 最大堆： 父节点的值大于或等于其子节点的值。
  堆常用于实现优先队列和堆排序等算法。

== 看到一直要用min，max的基本都要用堆

```python
import heapq
x = [1,2,3,5,7]

heapq.heapify(x)
###将列表转换为堆。

heapq.heappushpop(heap, item)
##将 item 放入堆中，然后弹出并返回 heap 的最小元素。该组合操作比先调用 heappush() 再调用 heappop() 运行起来更有效率

heapq.heapreplace(heap, item)
##弹出并返回最小的元素，并且添加一个新元素item

heapq.heappop(heap,item)
heapq.heappush(heap,item)
```



### 懒删除

懒删除就是，我表面上删除一个元素，实际上没有从堆里拿出来。

而是当我访问堆，要pop堆顶的时候，检查一下这个元素被没被删过

http://cs101.openjudge.cn/2024sp_routine/22067/

```python
import heapq
stack = []
Min = []
min_pop = set()
while True:
    try:
        s = input()
        if s == 'pop':
            if stack:
                min_pop.add(stack.pop())
        elif s == 'min':
            while Min and Min[0] in min_pop:
                min_pop.remove(heapq.heappop(Min))
            if Min:
                print(Min[0])
        elif 'push' in s:
            stack.append(int(list(s.split())[-1]))
            heapq.heappush(Min,stack[-1])
    except EOFError:
        break
```

### 求中位数

构建==大根堆和小根堆==（这里大根堆用相反数构建，保证输入的数据都是恒正的）

因为只需要求中位数，所以只要注意一下两个堆元素之差不能大于1

```python
import heapq

n = int(input())

def insert(num):
    if len(Min) == 0 or num > Min[0]:
        heapq.heappush(Min, num)
    else:
        heapq.heappush(Max,-num)
    if len(Min) - len(Max) > 1:
        heapq.heappush(Max,-heapq.heappop(Min))
    elif len(Max) - len(Min) > 1:
        heapq.heappush(Min, -heapq.heappop(Max))
for _ in range(n):
    result = []
    count = 0
    Min = []
    Max = []
    lst = list(map(int,input().split()))
    for num in range(len(lst)):
        count += 1
        insert(lst[num])
        if count % 2 == 1:
            if len(Min) > len(Max):
                result.append(Min[0])
            else:
                result.append(-Max[0])
    print(len(result))
    print(*result)

```

## 栈

#### 波兰表达式

波兰表达式运算

**从右向左**遍历给定字符串，若遇到数字，则压入栈。则**越左边的数字就越接近栈顶**。当遍历到一个运算符，根据后序表达式的定义，需要计算该运算符右边的两个数字之间经过此运算后的结果，然后**将结果重新塞回原位继续计算剩下的表达式**。该运算符右边的两个数字就恰好是栈顶的和次栈顶的两个数字！最后输出栈内唯一剩下的数字，就是最终的结果。

```python
import math
operators={'+','-','*','/'}
l=input().split()
stack=[]
n=len(l)
for i in range(-1,-n-1,-1):
    letter=l[i]
    if letter not in operators:#数字
        stack.append(letter)#压入栈
    else:
        num1=stack.pop(-1)
        num2=stack.pop(-1)
        newnum=eval(str(num1)+letter+str(num2))
        stack.append(newnum)
print('{:.6f}'.format(stack[0]))
        
```

中序表达式转后序表达式

```python
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2}
    stack = []
    postfix = []
    number = ''

    for char in expression:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                postfix.append(int(num) if num.is_integer() else num)
                number = ''
            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char] <= precedence[stack[-1]]:
                    postfix.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()

    if number:
        num = float(number)
        postfix.append(int(num) if num.is_integer() else num)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(str(x) for x in postfix)

n = int(input())
for _ in range(n):
    expression = input()
    print(infix_to_postfix(expression))
```

#### FILO

入栈的顺序是降序排列（如6,5,4,3,2,1)，由于栈是FILO，那么出栈序列任意数A的后面比A大的数都是按照升序排列的；

入栈的顺序是升序排列（如1,2,3,4,5,6)，由于栈是FILO，那么出栈序列任意数A的后面比A小的数都是按照降序排列的。

## Dilworth定理:

Dilworth定理表明

**任何一个有限偏序集的最长反链(即最长下降子序列)的长度等于将该偏序集划分为尽量少的链(即上升子序列)的最小数量。**

跳高：http://cs101.openjudge.cn/2024sp_routine/28389/

因此，计算序列的最长下降子序列长度，即可得出最少需要多少台测试仪。



## 排序

```python
#归并排序
import sys
sys.setrecursionlimit(100000)
def merge(arr,l,m,r):
    '''对l到m和m到r两段进行合并'''
    n1=m-l+1#L1长
    n2=r-m#L2长
    L1=arr[l:m+1]
    L2=arr[m+1:r+1]
    ''' L1和L2均为有序序列'''
    i,j,k=0,0,l#i为L1指针，j为L2指针，k为arr指针
    '''双指针法合并序列'''
    while i<n1 and j<n2:
        if L1[i]<=L2[j]:
            arr[k]=L1[i]
            i+=1
        else:
            arr[k]=L2[j]
            j+=1
        k+=1
    while i<n1:
        arr[k]=L1[i]
        i+=1
        k+=1
    while j<n2:
        arr[k]=L2[j]
        j+=1
        k+=1
def mergesort(arr,l,r):
    '''对arr的l到r一段进行排序'''
    if l<r:#递归结束条件，很重要
        m=(l+r)//2
        mergesort(arr,l,m)
        mergesort(arr,m+1,r)
        merge(arr,l,m,r)
n=int(input())#序列长
array=[int(y) for y in input().split()]
mergesort(array,0,n-1)
print(array)
```

````python
#### 快速排序

```python
def quicksort(arr, left, right):
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)


def partition(arr, left, right):
    # 最右端元素作为基准元素，i,j是两个指针，通过两个元素的交换实现
    # 基准元素左右分别小于、大于他本身。
    i = left
    j = right - 1
    pivot = arr[right]
    while i <= j:
        while i <= right and arr[i] < pivot:
            i += 1
        while j >= left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i


arr = [22, 11, 88, 66, 55, 77, 33, 44]
quicksort(arr, 0, len(arr) - 1)
print(arr)

# [11, 22, 33, 44, 55, 66, 77, 88]
```
````

