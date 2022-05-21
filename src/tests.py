from linalg.vector import Vector
from linalg.matrix import Matrix
from linalg import eqn_solve, bestfit_line

x=Vector(1,2,3)
y=Vector(2,'3/2',fractional=True)
print(f"x={x}\nb={y}")
print(f'x.dim={x.dim}, y.dim={y.dim}')
print(f'x+y = {x+y}')
print(f'x-y = {x-y}')
print(f'x/2 = {x/2}')
print(f'2b = {2*y}')
print(f'x.y = {x*y}')
print(f'x x y = {x@y}')
print(f'|x| = {x.norm()}')
print(f'angle(x,y) = {Vector.angle(x,y)}')
print(f'Projection of x onto y: {x.project_on(y)}')
print(f'x^ = {x.unit()}')
print(f'x%2 = {x%2}')
print(f'x^11 = {x**11}')
print(f'(x^100)%995 = {pow(x,100,995)}')

print('\n'*3)

a=Matrix([
    [5,7,9],
    [4,3, 8],
    [7,5,6]])

b=Matrix(['22/7','3.5','5'],fractional=True)


print(a)
print(a.dim,a.det,a.trace())
print(a.inverse())
print(a.ref())
print(a.rref())
print(a.is_square(),a.is_diagonal())
c,d=a.ref(),a.rref()
print(c,d,sep='\n')
print(c.is_ref(),c.is_rref())
print(d.is_ref(),d.is_rref())
print(c.is_upper_triangular(),c.is_lower_triangular(),d.is_diagonal())
print(a.lower_upper())
print(b)
print(b.T)
print(b*14)
print(a*b)

e=Matrix([[1,2,3,4],[1,0,1,0],[5,3,2,4],[6,1,4,6]])
f=Matrix([[1,2],[4,3],[5,6],[8,7]])
print(eqn_solve(e,f))

a=Matrix([10,15,20,25,30]).concat(Matrix([1]*5))
b=Matrix([13,18,24,27,31])
print(bestfit_line(a,b))
