# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 00:03:39 2015

Comp409 assignment1 Gradient Domain Mesh Editing

@author: Wentao Kuang
studentId:300314565

Total principle: L.V=d. L:Laplacian Matrix. V:Three dimension vertices Matrix. d: Gradeint Domain Matrix.
1.build the L and V(original vertices) and get the d.
2.build the constraints of Target(moving) vertices and constraints of preserving(anchor) vertices, and stack to L and d.
3.solve L.V=d with normal equation to get the new vertices.


ref:http://ecs.victoria.ac.nz/foswiki/pub/Courses/COMP409_2015T2/LectureSchedule/alex_maya_python.pdf
"""
import sys
import maya.cmds as cmds
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
from scipy.sparse.linalg import spsolve



kPluginCmdName = "GDME"


class ShapeDefomation(OpenMayaMPx.MPxCommand):
    
    '''Initialize.'''  
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)        
        #self.createWindow()
        
    '''After loaded the pulgin and excute command of this pulgin will excute this function'''    
    def doIt(self,argList):
        self.createWindow()
        
    # I like to keep all the iportant UI elements in a dictionary.
    UIElements = {}    
    
    '''This function creates the window.''' 
    def createWindow(self):       
        self.UIElements['window'] = cmds.window()
        self.UIElements['main_layout'] = cmds.columnLayout( adjustableColumn=True )
        self.UIElements['objectname'] = cmds.text(label=('please input the mesh\'s name'))
        self.UIElements['path'] = cmds.textField('name',tx='goblin',fn='fixedWidthFont',bgc=(0,0,0))
        self.UIElements['file'] = cmds.text(label=('please input a location to save data'))
        self.UIElements['path'] = cmds.textField('input',tx='E:\\mayaInfo',fn='fixedWidthFont',bgc=(0,0,0))
        self.UIElements['setanchor'] = cmds.button( label=('SaveAnchors '), command=self.setAnchor )
        self.UIElements['polyTriangulate'] =cmds.button(label=('polyTriangulate'),command=self.setTriangulate)
        self.UIElements['reshapeUniform'] = cmds.button( label=('Reshape(Uniform)'), command=self.reshapeUniform ) 
        self.UIElements['reshapeCotangent'] = cmds.button( label=('Reshape(Cotangent)'), command=self.reshapeContangent )
        cmds.showWindow( self.UIElements['window'] )

    '''
    This function listen the button SaveAnchors, 
    then save selected anchors and original vertices into corresponding location
    '''    
    def setAnchor(self ,*args):
        mayaInfo=cmds.textField('input',q=True,tx=True)
        name=cmds.textField('name',q=True,tx=True)                    
        selection = self.getMeshSelectionByName(name)
        self.saveInformation(mayaInfo,selection)
        
    def setTriangulate(self,*args):
        name=cmds.textField('name',q=True,tx=True)
        cmds.polyTriangulate(name) 
        
    '''
    This function listen the button Reshape,
    then excute getTargetVertices and set to new shape
    '''        
    def reshapeUniform(self, *args):
        mayaInfo=cmds.textField('input',q=True,tx=True)
        name=cmds.textField('name',q=True,tx=True)                
        selection = self.getMeshSelectionByName(name)
        TargetVertices=self.getTargetVertices(mayaInfo,selection,'uniform')
        self.setVertices(selection,TargetVertices)
        
    def reshapeContangent(self, *args):
        mayaInfo=cmds.textField('input',q=True,tx=True)
        name=cmds.textField('name',q=True,tx=True)                 
        selection = self.getMeshSelectionByName(name)
        TargetVertices=self.getTargetVertices(mayaInfo,selection,'cotangent')
        self.setVertices(selection,TargetVertices)

    '''Find the selection of MObjects'''
    def getMeshSelectionByName(self,name):
        try:
            selectionList = OpenMaya.MSelectionList()
            OpenMaya.MGlobal.getSelectionListByName(name, selectionList)
            selection = OpenMaya.MItSelectionList(selectionList,OpenMaya.MFn.kMesh)
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!please check the the inputed mesh name!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return selection
    
    '''
    Build the vertices matrix of a selected mesh
    '''    
    def getVertices(self,selection, space=OpenMaya.MSpace.kObject):
        # get dagPath
        dagPath = OpenMaya.MDagPath()
        selection.getDagPath(dagPath)
        mesh = OpenMaya.MFnMesh(dagPath)
        # create an empty point array
        point_array = OpenMaya.MPointArray()
        # get coordinates
        mesh.getPoints(point_array, space)
        # read to a numpy array
        num_vertices = point_array.length()
        vertices = np.zeros((num_vertices,3))
        for i in range(0, num_vertices):
            vertices[i,0] = point_array[i][0]
            vertices[i,1] = point_array[i][1]
            vertices[i,2] = point_array[i][2]
        return vertices 
    
    '''
    Set the matrix to vertices to corresponding mesh
    '''
    def setVertices(self,selection, vertices, space=OpenMaya.MSpace.kObject):
        # get dagPath
        dagPath = OpenMaya.MDagPath()
        selection.getDagPath(dagPath)
        mesh = OpenMaya.MFnMesh(dagPath)
        # create an empty point array
        point_array = OpenMaya.MPointArray()
        # allocate
        point_array.setLength(vertices.shape[0])
        for i in range(0, point_array.length()):
            point_array.set(i,vertices[i,0],vertices[i,1],vertices[i,2])
            # set coordinates
        mesh.setPoints(point_array, space)
        
    '''
    set colors to vertex
    '''
    def setVertexColor(self,selection, colors):
        dagPath = OpenMaya.MDagPath()
        selection.getDagPath(dagPath)
        mesh = OpenMaya.MFnMesh(dagPath)
        num_vertices = colors.shape[0]
        vc = OpenMaya.MColorArray()
        vc.setLength(num_vertices)
        for i in range(0, num_vertices):
            vc.set(i,colors[i,0],colors[i,1],colors[i,2])
            vi = OpenMaya.MIntArray()
            vi.setLength(num_vertices)
        for i in range(0, num_vertices):
            vi.set(i,i)
        mesh.setVertexColors(vc, vi) 
            
    '''
    Return a vertex iterator in the selection
    '''
    def getVertiexIterator(self,selection):
        dagPath=OpenMaya.MDagPath()
        selection.getDagPath(dagPath)
        mesh=OpenMaya.MFnMesh(dagPath)
        vertexs=OpenMaya.MItMeshVertex(mesh.object())
        return vertexs 
    
    '''
    purpose: get the selected vertices 
    '''
    def getSelectedVertices(self):
        try:
            selection=OpenMaya.MSelectionList()
            maya_selection=OpenMaya.MRichSelection()
            OpenMaya.MGlobal.getRichSelection(maya_selection)
            maya_selection.getSelection(selection)
            dagPath=OpenMaya.MDagPath()
            component=OpenMaya.MObject()
            iter=OpenMaya.MItSelectionList(selection,OpenMaya.MFn.kMeshVertComponent)
            elements=[]
            while not iter.isDone():
                iter.getDagPath(dagPath,component)
                dagPath.pop()
                node=dagPath.fullPathName()
                fnComp=OpenMaya.MFnSingleIndexedComponent(component)
                for i in range(fnComp.elementCount()):
                    elements.append(fnComp.element(i))
                iter.next()
        except:
            elements=[0]            
        return elements

    ''' generate a matrix of all the neighbours based on the number of vertices '''  
    def getNeighbours(self,selection):
        vertexIt=self.getVertiexIterator(selection)
        vertexConnects=OpenMaya.MIntArray()
        neighbours=[]
        vertexIt.reset()
        for i in range(vertexIt.count()):
            vertexIt.getConnectedVertices(vertexConnects)
            neighbours.append(list(vertexConnects))
            vertexIt.next()
        return neighbours

            
    '''
    build the laplace matrix with uniform weight
    principle:
    1. build a sparse matrix with the size of n*n, n is the number of vertices
    2. set the digonal with the minus number of neibours of corresponding vertice(Degree Matrix).
    3. set the each vertice's neighbour to its column with 1.(Adjacency Matrix)
    '''
    def setLaplaceMatrix(self,selection):
        vertices=self.getVertiexIterator(selection)
        Num=vertices.count()
        laplacianMatrix=sp.sparse.lil_matrix((Num,Num))
        Neighbours=OpenMaya.MIntArray()
        for i in range(0,Num):
            vertices.getConnectedVertices(Neighbours)
            laplacianMatrix[i,i]=len(Neighbours)*(-1.0)
            for j in range(0,len(Neighbours)):
                laplacianMatrix[i,Neighbours[j]]=1.0
            vertices.next()
        #print(laplacianMatrix) 
        return  laplacianMatrix
    
    '''
    get the gradient domain of the original vertices
    principle:
    1.Build the n*n Laplacian matrix first, L.
    2.get the original vertices into a n*3 matrix, v.
    3.get the gradient domain by d=L.v
    '''
    def getGradienDomain(self,Laplacian,OVertices):
        Laplacian=Laplacian.tocsr()
        gradientDomain=Laplacian.dot(sparse.csr_matrix(OVertices))
        #print(gradientDomain)
        return gradientDomain
    
    '''save the original vertices, and anchor vertices into path mayaInfo.'''    
    def saveInformation(self,mayaInfo,selection):
        print(mayaInfo)
        try: 
            Overtices=self.getVertices(selection)
            vAnchors=self.getSelectedVertices()
            np.savez(mayaInfo, Overtices=Overtices, vAnchors=vAnchors)
            print('Information saved successfully, with the location:'+mayaInfo)
        except:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Can not save the information! Please check and change the mayaInfo in order to save the information!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    '''original vertices and anchor vertices from path mayaInfo.''' 
    def loadInformation(self,mayaInfo):
        information=np.load(mayaInfo+'.npz')  
        OVertices=information['Overtices']
        vAnchors=information['vAnchors']
        return OVertices,vAnchors
    
    '''    
    build the constraints matrix of the target vertices  
    principle:
    1.build the constraints of the linear system L.V=d
    2.L constraints is a NL*Nv matrix, NL is the size of L, Nv is the number of selected target vertices.
    3.For L constraints, set the column number match to vertice with 1.
    4.d constraints is a NL*3 matrix, set the x,y,z of target vertices.
    '''  
    def setTargetVerticeConstraint(self,selection):
        try:
            Selectedvertex= self.getSelectedVertices()
        except:
            Selectedvertex=[0]
        Vertices=self.getVertices(selection)
        LaplacianTarget=sparse.lil_matrix((len(Selectedvertex),len(Vertices)))
        GradientTarget=np.zeros((len(Selectedvertex),3))
        for i in range(0,len(Selectedvertex)):     
            LaplacianTarget[i,Selectedvertex[i]]=1                      
            GradientTarget[i]=Vertices[Selectedvertex[i]]
        #print(GradientTarget)
        return LaplacianTarget,GradientTarget
    
    '''
    build the constraint matrix of the preserving(anchor) vertices
    principle:
    1.The same to set target vertices. Just replace the selected target vertices to selected anchor vertices.
    '''
    def setAnchorsConstraint(self,OVertices,vAnchors):
        try:        
            LaplacianAnchor=sparse.lil_matrix((len(vAnchors),len(OVertices))) 
            GradientAnchor=np.zeros((len(vAnchors),3))
            for i in range(0,len(vAnchors)):
                LaplacianAnchor[i,vAnchors[i]]=1
                GradientAnchor[i]=OVertices[vAnchors[i]]        
        except:
            LaplacianAnchor=''
            GradientAnchor=''
        return LaplacianAnchor,GradientAnchor

    '''get the other public(shared) neighbours of two neighbour vertices'''
    def getPublicNeighbours(self,iNeighbours,jNeighbours):
        pNeighbours=set(iNeighbours)&set(jNeighbours)
        return list(pNeighbours)

    '''compute the length of edge between two vertices'''
    def distance(self,i,j):
        distance=np.sqrt((i[0]-j[0])**2+(i[1]-j[1])**2+(i[2]-j[2])**2)
        return distance
        
    '''compute the contangent based on three egdes'''     
    def getCotangent(self,i,j,k):
        ij=self.distance(i,j)
        ik=self.distance(i,k)
        jk=self.distance(j,k)    
        a=(ik**2+jk**2-ij**2)/(2.0*jk)
        b=np.sqrt(ik**2-a**2)
        cot=a/b
        return cot
        
    '''compute the contangent based on three egdes'''
    def cot(self,i,j,k):
        ij=self.distance(i,j)
        ik=self.distance(i,k)
        jk=self.distance(j,k)
        r=(ij+ik+jk)
        Aijk=np.sqrt(r*(r-ij)(r-ik)(r-jk))
        cot=(ik**2+jk**2-ij**2)/4.0*Aijk
        return cot
    
    '''set a matrix of contangent weights'''
    def getCotangentWeight(self,vtxi,neighbour,vertices):
        iNeighbour=neighbour[vtxi]
        weights=[]
        for i in iNeighbour:
            pNeighbours=self.getPublicNeighbours(iNeighbour,neighbour[i])
            cot1=self.getCotangent(vertices[vtxi],vertices[i],vertices[pNeighbours[0]])
            cot2=self.getCotangent(vertices[vtxi],vertices[i],vertices[pNeighbours[1]])
            weight=abs(cot1+cot2)/2
            weights.append(weight)
            sumWeights = sum(weights)
        for i in range(0,len(weights)):
            weights[i]=weights[i]/sumWeights
        return weights 
    
    '''
    build the laplace matrix with cotangent weight
    principle:
    1. build a sparse matrix with the size of n*n, n is the number of vertices
    2. set the digonal with the -1 (Degree Matrix).
    3. set the each vertice's neighbour to its column with contangent weight.(Adjacency Matrix)
    '''    
        
    def setContangentLaplaceMatrix(self,selection):
        try:
            verticesIt=self.getVertiexIterator(selection)
            Num=verticesIt.count()
            laplacianMatrix=sp.sparse.lil_matrix((Num,Num))
            Neighbours=self.getNeighbours(selection)
            vertices=self.getVertices(selection)
            for i in range(0,len(Neighbours)):
                laplacianMatrix[i,i]=-1.0
                Weights=self.getCotangentWeight(i,Neighbours,vertices)
                for j in range(0,len(Neighbours[i])):
                    laplacianMatrix[i,Neighbours[i][j]]=Weights[j]
        except:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Please Triangulate the mesh first')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return laplacianMatrix
    
    
    
        
    '''
    get the reshaped vertices by sovling L.V=d use normal equation
    principle:
    1.stack target constraints and anchor constraints to Laplacian matrix and gradien domain matrix.
    2.Then sovle L.V=d with normal equation. (L.T.L).V=(L.T.d) to get the new V
    '''  
    def getTargetVertices(self,mayaInfo,selection,weight):
        if weight=='uniform':
            Laplacian=self.setLaplaceMatrix(selection)
        elif weight=='cotangent':
            Laplacian=self.setContangentLaplaceMatrix(selection)
        OVertices,vAnchors=self.loadInformation(mayaInfo)
        GradienDomain=self.getGradienDomain(Laplacian,OVertices)
        LaplacianTarget,GradientTarget=self.setTargetVerticeConstraint(selection)
        LaplacianAnchor,GradientAnchor=self.setAnchorsConstraint(OVertices,vAnchors)
        Laplacian_T=sparse.vstack([Laplacian,LaplacianTarget])
        GradienDomain_T=sparse.vstack([GradienDomain,GradientTarget])
        if LaplacianAnchor!='' and GradientAnchor!='':
            Laplacian_T_A=sparse.vstack([Laplacian_T,LaplacianAnchor])    
            GradienDomain_T_A=sparse.vstack([GradienDomain_T,GradientAnchor])            
        TargetVertices=spsolve(Laplacian_T_A.T.dot(Laplacian_T_A),Laplacian_T_A.T.dot(GradienDomain_T_A))
        return  TargetVertices

        
# End of class

'''Creator'''
def cmdCreator():
    return OpenMayaMPx.asMPxPtr( ShapeDefomation() )
    
'''Initialize the script plug-in'''
def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    try:
        mplugin.registerCommand( kPluginCmdName, cmdCreator )
    except:
        sys.stderr.write( "Failed to register command: %s\n" % kPluginCmdName )
        raise

'''Uninitialize the script plug-in'''
def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand( kPluginCmdName )
    except:
        sys.stderr.write( "Failed to unregister command: %s\n" % kPluginCmdName )

demo = ShapeDefomation()
