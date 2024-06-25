# Assignment #A: 图论：算法，树算及栈

Updated 2018 GMT+8 Apr 21, 2024

2024 spring, Complied by ==卢卓然 生命科学学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS 

Python编程环境：vscode

C/C++编程环境：无



## 1. 题目

### 20743: 整人的提词本

http://cs101.openjudge.cn/practice/20743/



思路：

这个函数通过使用栈来跟踪和反转括号内的字符。当遇到闭合括号 `)` 时，它会从栈中弹出字符直到遇到开放括号 `(`，然后将这些字符反转并放回栈中。最后，函数返回栈中剩余字符组成的字符串。

代码

```python
def reverse_parentheses(s):
    stack = []
    for char in s:
        if char == ')':
            temp = []
            while stack and stack[-1] != '(':
                temp.append(stack.pop())
            # remove the opening parenthesis
            if stack:
                stack.pop()
            # add the reversed characters back to the stack
            stack.extend(temp)
        else:
            stack.append(char)
    return ''.join(stack)

s = input().strip()
print(reverse_parentheses(s))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240430234653801](/Users/luzhuoran/Desktop/数算/作业/A/image-20240430234653801-4492017.png)



### 02255: 重建二叉树

http://cs101.openjudge.cn/practice/02255/



思路：

`build_tree` 函数是用来根据二叉树的前序遍历和中序遍历结果来重建该二叉树，并输出后序遍历结果的。这个函数的工作原理是递归地将前序遍历和中序遍历的列表分割成左右子树的对应部分，然后组合它们来构建整个树的后序遍历。

代码

```python
def build_tree(preorder, inorder):
    if not preorder:
        return ''
    
    root = preorder[0]
    root_index = inorder.index(root)
    
    left_preorder = preorder[1:1 + root_index]
    right_preorder = preorder[1 + root_index:]
    
    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]
    
    left_tree = build_tree(left_preorder, left_inorder)
    right_tree = build_tree(right_preorder, right_inorder)
    
    return left_tree + right_tree + root

while True:
    try:
        preorder, inorder = input().split()
        postorder = build_tree(preorder, inorder)
        print(postorder)
    except EOFError:
        break
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240430234739982](/Users/luzhuoran/Desktop/数算/作业/A/image-20240430234739982-4492061.png)



### 01426: Find The Multiple

http://cs101.openjudge.cn/practice/01426/

要求用bfs实现



思路：

 `find_multiple` 函数使用宽度优先搜索（BFS）算法来找到一个最小的由 “0” 和 “1” 组成的数字，这个数字可以被给定的 `n` 整除。这个算法使用队列来存储当前的数字和它们对 `n` 的模，同时检查每个数字是否能被 `n` 整除。

代码

```python
from collections import deque

def find_multiple(n):
    q = deque()
    q.append((1 % n, "1"))
    visited = set([1 % n])  

    while q:
        mod, num_str = q.popleft()
        if mod == 0:
            return num_str

        for digit in ["0", "1"]:
            new_num_str = num_str + digit
            new_mod = (mod * 10 + int(digit)) % n

            if new_mod not in visited:
                q.append((new_mod, new_num_str))
                visited.add(new_mod)

def main():
    while True:
        n = int(input())
        if n == 0:
            break
        print(find_multiple(n))

if __name__ == "__main__":
    main()
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240430234904646](/Users/luzhuoran/Desktop/数算/作业/A/image-20240430234904646-4492145.png)



### 04115: 鸣人和佐助

bfs, http://cs101.openjudge.cn/practice/04115/



思路：

- 初始化一个队列 `queue` 和一个集合 `pos` 来存储已经访问过的位置和查克拉状态。
- 将 `naruto` 的起始位置和初始查克拉加入队列。
- 使用四个方向数组 `stepx` 和 `stepy` 来表示上下左右移动。
- 在bfs函数中，循环直到队列为空：
  - 从队列中取出当前位置和查克拉状态。
  - 如果当前位置是 `sasuke` 的位置，则记录当前的深度（步数）并返回。
  - 否则，遍历四个可能的移动方向：
    - 如果新位置在网格范围内且是空地（`*`），并且没有在 `pos` 集合中，则将新位置和当前查克拉状态加入队列和 `pos` 集合。
    - 如果新位置是障碍（`#`）且查克拉大于0，并且新位置和减少后的查克拉状态没有在 `pos` 集合中，则将新位置和减少后的查克拉状态加入队列和 `pos` 集合。
- 最后，如果 `ans` 集合不为空，则打印最小的步数；如果为空，则打印 `-1` 表示无法到达。

请注意，这个算法假设 `naruto` 可以在没有障碍的情况下自由移动，并且每次移动到障碍时都会消耗一个查克拉。如果 `naruto` 的查克拉用完，则无法再通过障碍。这个算法的目标是找到最少步数的路径。如果有多条路径，则选择步数最少的一条。

代码

```python
from collections import deque

m, n, chakra = map(int, input().split())
graph = []
pos = set()
ans = []

naruto = sasuke = None
for i in range(m):
    arr = list(input())
    if '@' in arr:
        naruto = (i, arr.index('@'))
        pos.add((naruto[0], naruto[1], chakra))
    if '+' in arr:
        sasuke = (i, arr.index('+'))

    graph.append(arr)

graph[sasuke[0]][sasuke[1]] = '*'

stepx = [-1, 1, 0, 0]
stepy = [0, 0, -1, 1]


def bfs():
    global ans
    queue = deque()
    queue.append((naruto[0], naruto[1], 0, chakra))


    while queue:
        x, y, depth, chak = queue.popleft()

        if (x, y) == sasuke:
            ans.append(depth)
            return


        for i in range(4):
            x1, y1 = x + stepx[i], y + stepy[i]
            if 0 <= x1 < m and 0 <= y1 < n :
                if graph[x1][y1] == '*' and (x1, y1, chak) not in pos:
                    queue.append((x1, y1, depth + 1, chak))
                    pos.add((x1, y1, chak))

                elif chak > 0 and graph[x1][y1] == '#' and (x1, y1, chak-1) not in pos:
                    queue.append((x1, y1, depth + 1, chak-1))
                    pos.add((x1, y1, chak-1))

bfs()
if ans:
    print(min(ans))
else:
    print(-1)# 
# 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240430234949602](/Users/luzhuoran/Desktop/数算/作业/A/image-20240430234949602-4492190.png)



### 20106: 走山路

Dijkstra, http://cs101.openjudge.cn/practice/20106/



思路：dijkstra的标准题目！



代码

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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240430235040334](/Users/luzhuoran/Desktop/数算/作业/A/image-20240430235040334-4492241.png)



### 05442: 兔子与星空

Prim, http://cs101.openjudge.cn/practice/05442/



思路：

普里姆算法（Prim’s algorithm）是一种用于在加权无向图中找到最小生成树（MST）的贪心算法。最小生成树是图中包含所有顶点的边的子集，且这些边的总权重尽可能小。

普里姆算法的工作原理如下：

1. 从图中任意选择一个顶点作为最小生成树的起始点。
2. 维护两个顶点集合：已经包含在MST中的顶点集合和尚未包含在MST中的顶点集合。
3. 在每一步中，考虑连接这两个集合的所有边，并从中选择权重最小的边。
4. 将这条边的另一个端点移动到包含MST的顶点集合中。
5. 重复步骤3，直到所有顶点都包含在MST中。

普里姆算法的关键点在于，它始终保持一个顶点集合已经包含在MST中，然后逐步扩展这个集合，直到包含图中的所有顶点。在每一步中，算法都会选择连接已有MST和新顶点的最小权重边，从而保证了生成树的总权重最小。

代码

```python
import heapq

def prim(graph, start):
    mst = []
    used = set([start])
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))

    return mst

def solve():
    n = int(input())
    graph = {chr(i+65): {} for i in range(n)}
    for i in range(n-1):
        data = input().split()
        star = data[0]
        m = int(data[1])
        for j in range(m):
            to_star = data[2+j*2]
            cost = int(data[3+j*2])
            graph[star][to_star] = cost
            graph[to_star][star] = cost
    mst = prim(graph, 'A')
    print(sum(x[2] for x in mst))

solve()

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240430235122561](/Users/luzhuoran/Desktop/数算/作业/A/image-20240430235122561-4492283.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

本次作业以图的搜索为主，并不只是基础的bfs和dfs了，而是基于两种搜索的更高深的算法。

