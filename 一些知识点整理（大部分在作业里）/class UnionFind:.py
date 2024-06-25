class UnionFind:
    def __init__(self,n):
        self.parent=list(range(n))
        self.rank=[0]*n
    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,x,y):
        rootX=self.find(x)
        rootY=self.find(y)
        if rootX!=rootY:
            if self.rank[rootX]>self.rank[rootY]:
                self.parent[rootY]=rootX
            elif self.rank[rootX]<self.rank[rootY]:
                self.parent[rootX]=rootY
            else:
                self.parent[rootY]=rootX
                self.rank[rootX]+=1
    def joined(self,a,b):
        return self.find(a)==self.find(b)
n=int(input())
nums=set()
data=[]
for _ in range(n):
    s=input()
    a,key,b=s[0],s[1],s[3]
    data.append((a,key,b))
    nums.add(a)
    nums.add(b)
N=len(nums)
l=list(nums)
dic=dict()
for index,num in enumerate(l):
    dic[num]=index
uf=UnionFind(N)
notset=set()
flag=True
for d in data:
    x1,keys,y1=d[0],d[1],d[2]
    x1=dic[x1]
    y1=dic[y1]
    if keys=='=':
        if (x1,y1) in notset or (y1,x1) in notset:
            flag=False
        else:
            uf.union(x1,y1)
    else:
        notset.add((x1,y1))
        notset.add((y1,x1))
        if uf.joined(x1,y1):
            flag=False
if flag:
    print(True)
else:
    print(False)