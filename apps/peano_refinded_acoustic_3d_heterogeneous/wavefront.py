"""
    module wavefront

    Module to parse wavefront *.obj files

    Not all functionalities are implemented yet.
    Only the most useful (for us) mesh-related commands are recognized.
    
    Author : Alexis Mignon
    email  : alexis.mignon@info.unicaen.fr
    Date   : 01/03/2010

"""

import numpy as np

class Mesh(object) :
    def __init__(self,vertices = [],faces = [] ,uvcoords = [] , uvfaces  = [], groups = {}, normals = [], mtllib = None, name = ""):
        self.name = name
        self.vertices = vertices
        self.faces = faces
        self.uvcoords = uvcoords
        self.uvfaces = uvfaces
        self.groups = groups
        self.normals = normals
        self.mtllib = mtllib
 
    def translate(self,translation):
        vertices = np.array(self.vertices)
        vertices += translation
        self.vertices = vertices.tolist()
    
    def transform(self,transformation):
        """
            Apply a transformation to each vertex.
            if T is the transformation matrix,
            and V the column-vector representing the coordinates
            of the vertex. Then the transformed vector V' is
            V' = T*V.
            
            transformation matrix can be 3x3 or 4x4 for homogenous
            coordinates.
        """
        
        if transformation.shape == (3,3) :        
            self.vertices = np.dot(self.vertices,transformation.T).tolist()
        
        elif transformation.shape == (4,4) :
            vertices = np.ones( (len(self.vertices),4) )
            vertices[:,:-1] = self.vertices
            vertices = np.dot(vertices,transformation.T)
            vertices[:,:-1]/=vertices[:,-1]
            self.vertices = vertices[:,:-1].tolist()
        else : raise ValueError("Invalid shape for the transformation matrix (should be 3x3 or 4x4")



class Group(object) :
     def __init__(self,name,faces = [],material = None):
         self.name = name
         self.faces = faces
         self.material = material

def read_obj(filename):
    """
        Read an *.obj file and returns the corresponding objects.
    """
    group_name = None
    group = None
    object_name = None
    object = None
    mtllib = None
    nobjects = 0
    objects = []
    face_id = 0
    with open(filename) as f :
        for line in f.readlines() :
            if line.startswith('o '): # New object
                object_name = line.split()[1]
                object = None
            if   line.startswith('v '): # New vertex
                if object is None :
                    if object_name is not None : # if a name was given
                        object = Mesh(name = object_name)
                    else :
                        object = Mesh()
                    objects.append(object)
                if mtllib is not None :
                    object.mtllib = mtllib
                    mtllib = None
                object.vertices.append(map(float,line.split()[1:]))
                
            elif line.startswith('vt '):  # texture vertex
                object.uvcoords.append(map(float,line.split()[1:]))
                
            elif line.startswith('vn '):  # vertex normal
                object.normals.append(map(float,line.split()[1:]))
                
            elif line.startswith('f '):   # face
                indices = line.split()[1:]
                if len(indices) != 3 : continue
                if line.find('//')!=-1 : # we also have uv faces information
                    coords = [ map(lambda x : int(x) -1 ,i.split('//')) for i in indices ]
                    object.faces.append( [ c[0] for c in coords ] )
                    object.uvfaces.append( [c[1] for c in coords ] )
                elif line.find('/')!=-1 :
                    coords = [ map(lambda x : int(x) -1,i.split('/')) for i in indices ]
                    object.faces.append( [ c[0] for c in coords ] )
                    object.uvfaces.append( [c[1] for c in coords ] )                    
                else :
                    object.faces.append( map(lambda x : int(x) -1,line.split()[1:]) )

                if group is not None : group.faces.append(face_id)
                else : print "coucou"
                face_id += 1
    
            elif line.startswith('g ') : # new group
                group_name = line.split()[1]
                group = Group(name = gname)
                object.groups[group_name] = group
                
            elif line.startswith('usemtl ') : # material for the group
                if group is None :             # no group is defined
                    group_name = 'all_faces'   # so the material is for the all object
                    group = Group(name = group_name)
                    object.groups[group_name] = group
                group.material = line.split()[1]

            elif line.startswith('mtllib '): # the material file to use
                mtllib = line.split()[1]

    
    #if len(objects) == 1 : return object
    #else : 
    return objects
