from linalg.vector import Vector
from linalg.matrix import Matrix, DimensionError
from linalg import eqn_solve, bestfit_line


print('------ Vector ------')
x=Vector(1,2,3)
y=Vector(2,'3/2',fractional=True)
print(f"x={x} , b={y}")
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

print('\n'*5)

print('------ Matrix ------')
a=Matrix([
    [5,7,9],
    [4,3, 8],
    [7,5,6]])

b=Matrix(['22/7','3.5','5'],fractional=True)


print(a)
print('Dimension,Determinant,Trace')
print(a.dim,a.det,a.trace())
print(f'a.is_square: {a.is_square()}, a.is_diagonal: {a.is_diagonal()}')
print('A^{-1} : ')
print(a.inverse()) # inverse
c,d=a.ref(),a.rref() # row echelon form and reduced row echelon form
print('Row echelon form: ')
print(c)
print('Reduced row echelon form: ')
print(d)
print('Verify REF and RREF')
print(c.is_ref(),c.is_rref())
print(d.is_ref(),d.is_rref())
print(c.is_upper_triangular(),c.is_lower_triangular(),d.is_diagonal())

print("LU Decompose")
LU=a.lower_upper()
print(LU) # LU Decomposition

print('Slicing')
print(a[:2,:1])
print('only row slicing')
print(a[1:2]) # equivalent to a.row_slice(1,2)
print('only col slicing')
print(a[:,1:]) # equivalent to a.col_slice(1,None)


print('Mapping a function to every elements')
print(a.map(lambda x:x**2/2))


print('Fractional calculations')
print(b)
print(b.T) # transpose
print(b*14) # scalar multiplication
print(a*b) # multiplication ( equivalent to a@b or a.multiply(b) )

print('power of matrix')
print(repr(a**5)) # power

e=Matrix([[1,2,3,4],[1,0,1,0],[5,3,2,4],[6,1,4,6]])
f=Matrix([[1,2],[4,3],[5,6],[8,7]])
print('Solve system of linear equations:')
print(eqn_solve(e,f)) # expected: [[1.63333,1.3 ],[-0.166667,0.5 ][ 2.36667,1.7 ][-1.85,-1.35]]


print('Best fit line using least square method: ')
x=[10,15,20,25,30]
y=[13,18,24,27,31]
print(bestfit_line(x,y)) # expected [[0.9],[4.6]]
