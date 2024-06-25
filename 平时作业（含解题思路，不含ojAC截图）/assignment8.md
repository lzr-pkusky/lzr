# Assignment #8: 图论：概念、遍历，及 树算

Updated 1919 GMT+8 Apr 8, 2024

2024 spring, Complied by ==卢卓然 生命科学学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS

Python编程环境：vscode

C/C++编程环境：



## 1. 题目

### 19943: 图的拉普拉斯矩阵

matrices, http://cs101.openjudge.cn/practice/19943/

请定义Vertex类，Graph类，然后实现



思路：计概方法——用邻接矩阵表示图，一列一列的改变A。

数算方法——定义Vertex和Graph类，然后定义构建拉普拉斯矩阵的函数。



代码

```python
n,m=map(int,input().split())
A=[[0]*n for _ in range(n)]#注意生成矩阵的方式
#生成矩阵A
for _ in range(m):
    i,j=map(int,input().split())
    A[i][j]=A[j][i]=1
#计算D-A,一列一列的改变A，不会影响结果
for column in range(n):
    sumup=0
    for line in range(n):
        sumup+=A[line][column]
        A[line][column]=-A[line][column]
    A[column][column]+=sumup
    
for lines in A:
    print(' '.join([str(x) for x in lines]))
```

```python
class Vertex:	
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

def constructLaplacianMatrix(n, edges):
    graph = Graph()
    for i in range(n):	# 添加顶点
        graph.addVertex(i)
    
    for edge in edges:	# 添加边
        a, b = edge
        graph.addEdge(a, b)
        graph.addEdge(b, a)
    
    laplacianMatrix = []	# 构建拉普拉斯矩阵
    for vertex in graph:
        row = [0] * n
        row[vertex.getId()] = len(vertex.getConnections())
        for neighbor in vertex.getConnections():
            row[neighbor.getId()] = -1
        laplacianMatrix.append(row)

    return laplacianMatrix


n, m = map(int, input().split())	# 解析输入
edges = []
for i in range(m):
    a, b = map(int, input().split())
    edges.append((a, b))

laplacianMatrix = constructLaplacianMatrix(n, edges)	# 构建拉普拉斯矩阵

for row in laplacianMatrix:	# 输出结果
    print(' '.join(map(str, row)))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240416132449961](/Users/luzhuoran/Desktop/数算/作业/作业8/image-20240416132449961-3245093.png)



### 18160: 最大连通域面积

matrix/dfs similar, http://cs101.openjudge.cn/practice/18160



思路：思路是dfs，从一个点向四面八方dfs。visit数组用`matrix[x][y]='.'`替代，似乎更方便一点。



代码

```python
# 
dire = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]

area = 0
def dfs(x,y):
    global area
    if matrix[x][y] == '.':return
    matrix[x][y] = '.'
    area += 1
    for i in range(len(dire)):
        dfs(x+dire[i][0], y+dire[i][1])


for _ in range(int(input())):
    n,m = map(int,input().split())

    matrix = [['.' for _ in range(m+2)] for _ in range(n+2)]
    for i in range(1,n+1):
        matrix[i][1:-1] = input()

    sur = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if matrix[i][j] == 'W':
                area = 0 
                dfs(i, j)
                sur = max(sur, area)
    print(sur)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240416132612483](/Users/luzhuoran/Desktop/数算/作业/作业8/image-20240416132612483-3245173.png)



### sy383: 最大权值连通块

https://sunnywhy.com/sfbj/10/3/383



思路：学习题解的代码书写。DFS，从一个节点开始，访问它的每一个邻居。可以使用一个visited数组来跟踪每个节点是否已经被访问过。对于每个连通块，可以计算其权值之和，并更新最大权值。

有一点惊讶就是把def dfs写在了def max_weight函数的内部，这样写竟然是可以允许的。



代码

```python
def max_weight(n, m, weights, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    max_weight = 0

    def dfs(node):
        visited[node] = True
        total_weight = weights[node]
        for neighbor in graph[node]:
            if not visited[neighbor]:
                total_weight += dfs(neighbor)
        return total_weight

    for i in range(n):
        if not visited[i]:
            max_weight = max(max_weight, dfs(i))

    return max_weight

# 接收数据
n, m = map(int, input().split())
weights = list(map(int, input().split()))
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

print(max_weight(n, m, weights, edges))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240416133503461](/Users/luzhuoran/Desktop/数算/作业/作业8/image-20240416133503461-3245704.png)



### 03441: 4 Values whose Sum is 0

data structure/binary search, http://cs101.openjudge.cn/practice/03441



思路：首先创建一个字典来储存数对的数量和频率，然后找相加为0的组合。



代码

```python
from collections import defaultdict

def four_sum_count(A, B, C, D):
    # Create a hash map to store the sum of pairs and their frequencies
    sum_pairs = defaultdict(int)
    for a in A:
        for b in B:
            sum_pairs[a + b] += 1

    # Find the quadruplets with a sum of zero
    count = 0
    for c in C:
        for d in D:
            if -(c + d) in sum_pairs:
                count += sum_pairs[-(c + d)]

    return count
n=int(input())
A=[]
B=[]
C=[]
D=[]
for _ in range(n):
    a,b,c,d=map(int,input().split())
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)

# Compute the number of quadruplets
print(four_sum_count(A, B, C, D))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240416135115361](/Users/luzhuoran/Desktop/数算/作业/作业8/image-20240416135115361-3246678.png)



### 04089: 电话号码

trie, http://cs101.openjudge.cn/practice/04089/

Trie 数据结构可能需要自学下。



思路：

1. 创建一个名为 `Solution` 的类，其中包含一个名为 `is_consistent` 的方法。
2. 在 `is_consistent` 方法中，首先对电话号码列表进行排序。
3. 然后，遍历排序后的电话号码列表。对于每一对相邻的电话号码，检查是否存在一个电话号码是另一个电话号码的前缀。
4. 如果存在前缀关系，返回 `False`，否则返回 `True`。



代码

```python
class Solution:
    def is_consistent(self, phone_numbers):
        phone_numbers.sort()  
        for i in range(len(phone_numbers) - 1):
            if phone_numbers[i + 1].startswith(phone_numbers[i]):
                return False
        return True

def main():
    t = int(input().strip())
    for _ in range(t):
        n = int(input().strip())
        phone_numbers = [input().strip() for _ in range(n)]
        solution = Solution()
        if solution.is_consistent(phone_numbers):
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240416232433591](/Users/luzhuoran/Desktop/数算/作业/作业8/image-20240416232433591-3281076.png)



### 04082: 树的镜面映射

http://cs101.openjudge.cn/practice/04082/



思路：

1. 首先，代码定义了一个名为 `binarynode` 的类，用于表示树的节点。每个节点包含值、左子节点、右子节点、孩子列表和父节点。
2. 然后，从输入中读取树的节点数 `n` 和节点列表 `lst`。
3. 使用栈stack和节点列表nodes来构建树。遍历lst中的每个元素：
   - 如果当前元素表示内部节点（第二个字符为 “0”），则将其入栈。
   - 如果当前元素表示外部节点（第二个字符为 “1”），则从栈中弹出一个节点，并将当前节点作为其子节点。
4. 接下来，为每个节点设置父子关系。如果左子节点存在且不是虚节点（值不为 “$”），则将其添加到孩子列表中，并设置父节点。同样，如果右子节点存在且不是虚节点，则也设置父子关系。
5. 对于每个节点，将其孩子列表逆序，以实现镜面映射。
6. 最后，按照宽度优先遍历的顺序输出树的节点值。



代码

```python
class binarynode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.children = []
        self.parent = None

n = int(input())
lst = input().split()
stack = []
nodes = []
for x in lst:
    temp = binarynode(x[0])
    nodes.append(temp)
    if stack:
        if stack[-1].left:
            stack[-1].right = temp
            stack.pop()
        else:
            stack[-1].left = temp
    if x[1] == "0":
        stack.append(temp)

for x in nodes:
    if x.left and x.left.value != "$":
        x.children.append(x.left)
        x.left.parent = x
    if x.right and x.right.value != "$":
        x.parent.children.append(x.right)
        x.right.parent = x.parent

for x in nodes:
    x.children = x.children[::-1]

lst1 = [nodes[0]]
for x in lst1:
    if x.children:
        lst1 += x.children
print(" ".join([x.value for x in lst1]))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240416232808075](/Users/luzhuoran/Desktop/数算/作业/作业8/image-20240416232808075-3281289.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

与计概有相同也有不同，很多题目需要定义类、定义函数进行求解，这种方法看似麻烦，但似乎更体现了数算的特点。结合树相关的知识会有新的收获。

