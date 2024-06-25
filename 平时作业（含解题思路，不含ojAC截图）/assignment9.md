# Assignment #9: 图论：遍历，及 树算

Updated 1739 GMT+8 Apr 14, 2024

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

### 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



思路：根据dfs序列递归建树，但并不需要把树转换为实体二叉树，可以递归地求高度(Height)和转换后高度(NewH)。



代码

```python
class Node:
    def __init__(self):
        self.child = []
    
    def getHeight(self):
        return 1 + max([nd.getHeight() for nd in self.child], default=-1)
    
    def getNewH(self):
        return 1 + max([nd.getNewH() + i for i, nd in enumerate(self.child)], default=-1)

def call():
    res = Node()
    while s and s.pop() == 'd':
        res.child.append(call())
    return res
    
s = list(input())[::-1]
root = call()
print(f"{root.getHeight()} => {root.getNewH()}")
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240423224715213](/Users/luzhuoran/Desktop/数算/作业/9/image-20240423224715213-3883639.png)

### 08581: 扩展二叉树

http://cs101.openjudge.cn/dsapre/08581/



思路：

1. `Node` 类：
   - `__init__` 方法初始化一个节点，包含一个名字 (`name`) 和子节点列表 (`child`)。
2. `getSeq` 函数：
   - 这是一个递归函数，它遍历树的每个节点。
   - 对于每个节点，它会递归地调用自身来获取子节点的序列。
   - 然后，它会在指定位置 (`pos`) 插入当前节点的名字。
   - 最终，它返回连接起来的字符串，包含了从当前节点开始的序列。
3. `generateTree` 函数：
   - 这也是一个递归函数，用于根据输入构建树。
   - 它从输入字符串 `s` 的末尾开始，每次弹出一个字符作为节点的名字。
   - 如果字符不是 `'.'`，则创建一个新的 `Node`，并递归地为其左右子节点 (`'lr'`) 调用 `generateTree`。
4. 输入和输出处理：
   - 用户输入的字符串被反转并存储在 `s` 中。
   - 调用 `generateTree` 函数生成树的根节点。
   - 然后，循环调用 `getSeq` 函数两次，分别在位置 1 和 2 插入根节点的名字，打印出两种不同的序列。



代码

```python
class Node:
    def __init__(self, name, child):
        self.name = name
        self.child = child

def getSeq(node, pos):
    if node:
        sub = [getSeq(nd, pos) for nd in node.child]
        sub.insert(pos, node.name)
        return ''.join(sub)
    return ''

def generateTree():
    if (token := s.pop()) != '.':
        return Node(token, [generateTree() for i in 'lr'])

s = list(input())[::-1]
root = generateTree()
for i in range(1, 3):
    print(getSeq(root, i))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240423225740048](/Users/luzhuoran/Desktop/数算/作业/9/image-20240423225740048-3884261.png)



### 22067: 快速堆猪

http://cs101.openjudge.cn/practice/22067/



思路：

这段代码的思路是实现一个支持查询当前最小元素的栈。具体来说，它可以处理三种操作：

1. **插入操作**：当输入的字符串以数字结尾时，它会将该数字插入栈中。如果栈为空，直接插入该数字。否则，插入该数字和栈顶元素中较小的一个。
2. **最小值查询**：当输入为 `'min'` 且栈非空时，打印栈顶元素，即当前栈中的最小值。
3. **弹出操作**：当输入为 `'pop'` 且栈非空时，弹出栈顶元素。

这种方法的优点是可以在常数时间内查询到栈中的最小元素。这是因为每次插入操作时，栈中都会保存一个当前的最小值。当进行弹出操作时，由于每个元素都与一个最小值相关联，所以可以保持最小值的正确性。

代码

```python
minpig = []
while True:
    try:
        s = input()
    except EOFError:
        break
    if s == 'min' and minpig:
        print(minpig[-1])
    elif s == 'pop' and minpig:
        minpig.pop()
    elif s[-1].isdigit():
        p = int(s.split()[1])
        if not minpig:
            minpig.append(p)
        minpig.append(min(minpig[-1], p))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240423230122176](/Users/luzhuoran/Desktop/数算/作业/9/image-20240423230122176-3884483.png)



### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123



思路：

1. 首先，我们定义了一个名为 `direc` 的列表，其中包含了所有可能的移动方向。
2. 然后，我们进入循环，根据用户输入的整数执行多次操作。
3. 在每次循环中，我们读取四个整数：`n`、`m`、`x` 和 `y`。这些整数表示一个 `n × m` 的网格，以及起始位置 `(x, y)`。
4. 我们创建一个 `n × m` 的二维列表 `graph`，并将所有元素初始化为 `1`，表示节点可访问。
5. 初始化一个变量 `ans` 为 `0`，用于记录满足条件的路径数量。
6. 定义递归函数 `dfs`，遍历相邻节点并更新已访问的节点数量。
7. 在起始位置 `(x, y)` 调用 `dfs` 函数，并打印出 `ans` 的值。



代码

```python
direc = [(j*i, k*(3-i)) for i in [1,2] for j in [-1,1] for k in [-1,1]]
for o in range(int(input())):
    n, m, x, y = map(int, input().split())
    graph = [[1]*m for i in range(n)]
    ans = 0
    def dfs(x, y, num):
        graph[x][y] = 0
        if num == n*m:
            global ans
            ans += 1
            return
        for dx, dy in direc:
            if 0<=x+dx<n and 0<=y+dy<m and graph[x+dx][y+dy]:
                dfs(x+dx, y+dy, num+1)
                graph[x+dx][y+dy] = 1
    dfs(x, y, 1)
    print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240423230244990](/Users/luzhuoran/Desktop/数算/作业/9/image-20240423230244990-3884566.png)



### 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/



思路：参考题解，学习了用🪣辅助建立图的思路。![image-20240417165439600](/Users/luzhuoran/Desktop/数算/作业/9/image-20240417165439600.png)



代码

```python
from collections import defaultdict, deque
from itertools import permutations

bucket = defaultdict(list)
for o in range(int(input())):
    word = input()
    for i in range(4):
        label = list(word)
        label[i] = '_'
        bucket[''.join(label)].append(word)
graph = defaultdict(list)
for words in bucket.values():
    for a, b in permutations(words, 2):
        graph[a].append(b)
start, end = input().split()
q = deque([start])
pre = dict()
used = set(q)
def bfs():
    while q:
        word = q.popleft()
        for nex in graph[word]:
            if nex not in used:
                used.add(nex)
                pre[nex] = word
                if nex == end:
                    return True
                q.append(nex)
res = bfs()
if res:
    ans = [end]
    while ans[-1] in pre:
        ans.append(pre[ans[-1]])
    print(*ans[::-1])
else:
    print('NO')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240423230534526](/Users/luzhuoran/Desktop/数算/作业/9/image-20240423230534526-3884735.png)



### 28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/



思路：

**Warnsdorff’s Rule**（也称为**Warnsdorff优化**）是解决**骑士巡游问题**的一种启发式算法。

- **Warnsdorff’s Rule**：
  - 我们可以从棋盘上的任意初始位置开始。
  - 每次移动，我们总是选择一个相邻且未访问过的方格，该方格具有最小的可访问邻居数（即最少的未访问相邻方格）。
  - 这样，我们通过贪心地选择最优的下一步，希望能够更快地找到解决方案。
- **算法步骤**：
  1. 随机选择一个初始位置，并在该位置标记为第一步。
  2. 从第二步开始，根据 Warnsdorff’s Rule，选择下一步的位置。
  3. 重复步骤 2，直到访问了所有方格。
  4. 返回标记好的棋盘，每个方格上标记了骑士访问的步数。
- **应用**：
  - Warnsdorff’s Rule 可以用于解决骑士巡游问题，尤其在较小的棋盘上效果显著。例如，它可以在 1 秒内解决 64×64 的巡游或 16×16 的周游问题。
  - 然而，对于更大的棋盘（例如 24×24），该算法的性能可能会受到限制。

代码

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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240423230739496](/Users/luzhuoran/Desktop/数算/作业/9/image-20240423230739496-3884860.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

图的问题也不是典型的模板，它涉及到一些额外的优化技巧。例如，在词梯问题中，我们需要使用桶来预处理数据；而在骑士周游问题中，我们需要引入排序并调整深度优先搜索（DFS）的顺序。进阶算法！



