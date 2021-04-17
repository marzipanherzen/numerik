'''
Numerik I für Ingenieurwissenschaften
Projekt: Wirbelpartikel
Barnes Hut Algorithmus
speed up for random vortex method 

created 16/04/2021 by @V. Herbig
'''

class Branch:
    def __init__(self, xm, ym, D):

        self.xm = xm
        self.ym = ym
        self.D = D

        self.hasChild = False
        self.children = [None,None,None,None] # NW -+, SW --, SE +-, NE ++
        
        self.hasLeaf = False
        self.xLeaf = None
        self.yLeaf = None
        self.GLeaf = None

def insertPoint(x,y,G,branch):
    if not branch.hasChild and not branch.hasLeaf:
        branch.xLeaf = x
        branch.yLeaf = y
        branch.GLeaf = G
        branch.hasLeaf = True
    else:
        if branch.hasLeaf:
            if branch.xLeaf < branch.xm:
                if branch.yLeaf < branch.ym: # SW
                    branch.children[1] = Branch(branch.xm-branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,branch.children[1]))
                else: # NW
                    branch.children[0] = Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,branch.children[0])
            else:
                if branch.yLeaf < branch.ym: # SE
                    branch.children[2] = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,branch.children[2])
                else: # NE
                    branch.children[3] = Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,branch.children[3])
            branch.hasLeaf = False

        if x < branch.xm:
            if y < branch.ym:
                if branch.children[1] == None: # SW
                    branch.children[1] = Branch(branch.xm-branch.D/4,branch.ym-branch.D/4,branch.D/2)
                insertPoint(x,y,G,branch.children[1])
            else: # NW
                if branch.children[0] == None:
                    branch.children[0] = Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                insertPoint(x,y,G,branch.children[0])
        else:
            if y < branch.ym: # SE
                if branch.children[2] == None:
                    branch.children[2] = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                insertPoint(x,y,G,branch.children[2])
            else: # NE
                if branch.children[3] == None:
                    branch.children[3] = Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                insertPoint(x,y,G,branch.children[3])            
        branch.hasChild = True

        branch.xLeaf = x
        branch.yLeaf += y
        branch.GLeaf += G

    
                
