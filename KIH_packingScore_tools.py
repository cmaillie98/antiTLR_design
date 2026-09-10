#this custom python package is largely adapted from Weiyi Tang's original work, with modificaiton from Marco Mravic & Colleen Maillie   
import sys, numpy as np, time, collections
from prody import *
from helix_dimer_geometryV3 import *
from fitting_cylinder import *
from collections import defaultdict
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley


#### global varibles #####

subsegment_len_dict = {}
for i in np.arange(9, 38):
		if i in np.arange(9, 13):
			subsegment_len_dict[i] =  7
		elif i in np.arange(13, 28):
			subsegment_len_dict[i] =  10 
		elif i in np.arange(28, 37):
			subsegment_len_dict[i] =  12
			
sasa 	= ShrakeRupley(probe_radius=1.2)  #sasa probe_radius=1.2 ~H2O 

bbAtoms = ['N', 'C', 'O', 'CA', 'H', 'HA', '1HA', '2HA']

#### classes  #####
class Knob:			#atoms = ProDy atom groups
	def __init__( self, chain, resnum, atoms ):
		self.chain 				= chain
		self.resnum 			= resnum
		self.atoms 				= atoms
		self.knob_ID   			= '%s_%d' % (chain, resnum)
		self.dist_to_hole34cen 	= 100
		self.total_contacts 	= 0
		self.hole_resi_reached 	= [] 
		self.hole_atom_reached 	= 0
		self.dSASA 				= []
		self.dSASA34 			= [] 
		self.dSASAcen 			= -1 

	def __repr__(self):
		return self.atoms



class Hole:		#atoms = ProDy atom groups
	def __init__( self, atoms, chain, resi_set, proj_coords, chain_subseg ):
		self.numAtoms		= len(resi_set)
		self.chain 			= chain
		self.fullID  		= '%s_%s' % (chain, '-'.join([str(s) for s in resi_set ]) )
		self.chain_subseg 	= chain_subseg
		self.CAs	 		= atoms.select('ca')
		self.atoms	 		= atoms
		self.resi_set		= resi_set
		self.proj_coords 	= proj_coords
		self.knobs_by_contact = {}
		self.position2resi  = {}
		self.resi2position  = {}
		self.official_Knobs = {}
		self.partial_Knobs 	= {}
		self.dSASA_perRes_dict =  {}

		# ID based off residue at i+3 positions, even for 3 res holes
		# calculate center of 'hole' as center bettern residues at i+3 & i+4 positions
		# calculate vector emerging from hole, cross product of 3-4 & 1-7 vector
		#	for special case of 3aa holes, use cen-1 or cen-7 vectors
		if self.numAtoms == 3: 

			if resi_set[1] - resi_set[0] == 3:		
				self.id   		= '%s_%d' % (chain, resi_set[ 1 ])
				self.ca34	 	= atoms.select('ca resid %d %d' % (resi_set[1], resi_set[2]) )
				self.ca34cen	= calcCenter(self.ca34)	
				self.hole_vect 	= np.cross(self.CAs[2].getCoords() - self.CAs[1].getCoords(), self.ca34cen - self.CAs[0].getCoords())
				self.hole_vect	= self.hole_vect/ LA.norm(self.hole_vect)
				self.position2resi[3] = resi_set[1] 
				self.position2resi[4] = resi_set[2] 
				self.position2resi[0] = resi_set[0]
				#print(self.fullID, self.hole_vect*2 + self.ca34cen)

			else:
				self.id   		= '%s_%d' % (chain, resi_set[ self.numAtoms - 3 ])
				self.ca34	 	= atoms.select('ca resid %d %d' % (resi_set[0], resi_set[1]) )
				self.ca34cen	= calcCenter(self.ca34)	
				#print( self.CAs[1].getCoords() - self.CAs[0].getCoords(), self.CAs[2].getCoords() - self.ca34cen   )
				self.hole_vect 	= np.cross(self.CAs[1].getCoords() - self.CAs[0].getCoords(), self.CAs[2].getCoords() - self.ca34cen)
				self.hole_vect	= self.hole_vect/ LA.norm(self.hole_vect)
				#print(self.id, self.hole_vect*2 + self.ca34cen)
				self.position2resi[3] = resi_set[0] 
				self.position2resi[4] = resi_set[1]
				self.position2resi[7] = resi_set[2]

		else:
			self.id   			= '%s_%d' % (chain, resi_set[ self.numAtoms - 3 ])
			self.ca34	 		= atoms.select('ca resid %d %d' % (resi_set[1], resi_set[2]) )
			self.ca34cen		= calcCenter(self.ca34)	
			self.hole_vect 	= np.cross(self.CAs[2].getCoords() - self.CAs[1].getCoords(), self.CAs[3].getCoords() - self.CAs[0].getCoords())
			self.hole_vect	= self.hole_vect/ LA.norm(self.hole_vect)
			self.position2resi[3] = resi_set[1] 
			self.position2resi[4] = resi_set[2]
			self.position2resi[0] = resi_set[0]
			self.position2resi[7] = resi_set[3]


		self.hole_mark34=self.hole_vect*1.5 + self.ca34cen

		for k,v in self.position2resi.items():
			self.resi2position[v] = k

		#print(self.fullID,'\t', self.id,  self.chain_subseg )
	def print_hole_markers(self, dir_path='.'):
		self.holeMarker_filepath = os.path.join( os.path.abspath( dir_path ), 'holeMark_%s.pdb' % self.id)
		write_markAtoms( [ np.array(self.hole_mark34) ]  , self.holeMarker_filepath )
		return 0

	def delete_hole_marker_file(self, dir_path='.'):
		self.holeMarker_filepath = os.path.join( os.path.abspath( dir_path ), 'holeMark_%s.pdb' % self.id)
		os.remove(self.holeMarker_filepath)
		return 0

	def load_hole_marker_PRODY(self):
		if not os.path.exists(self.holeMarker_filepath):
			print ('\n\nERROR looking for file %f not found' % self.holeMarker_filepath)
			sys.exit()
		self.ca34cen_atomgroup = parsePDB(self.holeMarker_filepath)
		self.ca34cen_atomgroup.setChids([self.chain])
		return 0


	def checkSASA(self, knob):

		### calculate the self SASA of backbone with sidechains (biopython parser)
		writePDB('tmp_hole%s.pdb' % self.id, self.atoms)
		p = PDBParser(QUIET=1)
		hole = p.get_structure(self.id, 'tmp_hole%s.pdb' % self.id)
		sasa.compute(hole)
		os.remove('tmp_hole%s.pdb' % self.id)

		self.hole_self_SASA_dict, self.hole_self_SASA = {}, 0
		for resi in hole[0][self.chain].get_residues():
			#for atom in resi: 
			#	if atom.get_name() in bbAtoms:
			#		print (atom, atom.sasa)
			bbSASA = round( np.sum([atom.sasa for atom in resi if atom.get_name() in bbAtoms  ] ), 2 )
			self.hole_self_SASA_dict[ resi.get_full_id()[3][1] ] = bbSASA
			self.hole_self_SASA += bbSASA
			#print ( '\t', resi.get_full_id()[3][1] , round( bbSASA, 2) )

		### calculate the deltaSASA, sasa of hole backbone which is reduced by knob
		## 	   find the per residue & total SASA of the hole's backbone atoms

		#print ('\n check sasa for',knobID, 'to hole', self)
		knobID = '%s%d' % ( knob.chain, knob.resnum )

		self.hole_wKnobs = self.atoms + knob.atoms 
		writePDB('tmp_hole%s_knob_%s.pdb' % (self.id, knobID) , self.hole_wKnobs)
		hole_wKnob = p.get_structure( knobID, './tmp_hole%s_knob_%s.pdb' % (self.id, knobID) )
		sasa.compute( hole_wKnob )
		os.remove( 'tmp_hole%s_knob_%s.pdb' % (self.id, knobID) )

		self.dSASA_perRes_dict[knobID]  = {}
		for resi in hole_wKnob[0][self.chain].get_residues():
				#for atom in resi: 
				#	if atom.get_name() in bbAtoms:
				#		print (resi, atom, round(atom.sasa, 2) )

				hole_resnum = resi.get_full_id()[3][1]
				# subtract SASA for hole+knob from SASA of knob alone
				hole_only_SASA 	= self.hole_self_SASA_dict[ hole_resnum ]
				bbSASA 			= round( np.sum([atom.sasa for atom in resi if atom.get_name() in bbAtoms  ] ) , 2)
				dSASA_perRes 	= hole_only_SASA - bbSASA
				# subtract SASA for hole+knob from SASA of knob alone
				self.dSASA_perRes_dict[knobID][ hole_resnum ] = dSASA_perRes
				knob.dSASA.append(dSASA_perRes)
				#print ( '\t', knobID, hole_resnum , round( bbSASA, 1), round( hole_only_SASA , 1), round(dSASA_perRes) )
				# check if knob residue is 3 or 4, then add to array
				if self.resi2position[ hole_resnum ] in [3,4]:
					knob.dSASA34.append(dSASA_perRes)
		
		### calculate the deltaSASA of hole center marker cause by knob, i.e. atom marker between i+3 & i+4 residues 1.5 Å outwards from hole
		## 	   
		self.load_hole_marker_PRODY()

		self.holeCen 		= self.atoms.copy() + self.ca34cen_atomgroup.copy() 
		writePDB('tmp_holeCen%s.pdb' % self.id, self.holeCen)
		p = PDBParser(QUIET=1)
		hole = p.get_structure(self.id, 'tmp_holeCen%s.pdb' % self.id)
		sasa.compute(hole)
		os.remove('tmp_holeCen%s.pdb' % self.id)

		self.holeCen_self_SASA   = 0
		for resi in hole[0][self.chain].get_residues():
			for atom in resi: 
				if atom.get_name() == 'CX':
					#print (atom, atom.sasa)
					self.holeCen_self_SASA = atom.sasa

		self.holeCen_wKnobs = self.atoms.copy() + self.ca34cen_atomgroup.copy() + knob.atoms.copy() 
		writePDB('tmp_holeCenKnob%s.pdb' % self.id, self.holeCen_wKnobs)
		p = PDBParser(QUIET=1)
		hole = p.get_structure(self.id, 'tmp_holeCenKnob%s.pdb' % self.id)
		sasa.compute(hole)
		os.remove('tmp_holeCenKnob%s.pdb' % self.id)

		self.holeCen_dSASA   = 0
		for resi in hole[0][self.chain].get_residues():
			for atom in resi: 
				if atom.get_name() == 'CX':
					#print (atom, atom.sasa)
					self.holeCen_dSASA 	= self.holeCen_self_SASA - atom.sasa
					knob.dSASAcen 		= self.holeCen_dSASA
		#print ('cenMarker dSASA', knob.dSASAcen, self.holeCen_self_SASA, atom.sasa)

		return (np.sum(knob.dSASA), np.sum(knob.dSASA34), knob.dSASAcen)

	def __repr__(self):
		return self.id


####  functions  #####

def packing_score_2helix( pdb ):
	number_knobs, packing_scoreTotal = 0,0

	#  grab protein chains, alphabetically first one will be 'A', check if <9aa, quit if so
	#  move chain A to 0,0,0   later we'll orient it with z=0 vector
	chains 	= sorted( list( set( pdb.getChids() ) ) )

	if len(chains) != 2:
		print('\n\tIncorrect number of chains. 2 accepted here. Quitting...')
		sys.exit()

	chain_A =  pdb.select('chain %s' % chains[0]) 
	chain_B =  pdb.select('chain %s' % chains[1])

	chain_A_CAs, chain_B_CAs = chain_A.select('ca'), chain_B.select('ca')
	len_chA, len_chB = len(chain_A_CAs), len(chain_B_CAs) 

	if len_chA < 9 or len_chB < 9:
		print('\n\tInput chain(s) too short (<9aa), A: %daa B: %daa. Quitting...' % (len(chain_A_CAs)), len(chain_B_CAs))
		sys.exit()
	elif len_chA  > 36 or len_chB > 36:
		print('\n\tInput chain(s) too short (>36aa), A: %daa B: %daa. Quitting...' % (len(chain_A_CAs)), len(chain_B_CAs))
		sys.exit()

	### Split each chain into 3 continguous segments. We will fit a helical axis vector through each subset
	 # Defining small helical sub-segments allows for distinct local geometry of each segment (i.e. kinks,curvature)
	 # vs 1 helix axis vector across entire TM helix span.  Use vector to help define inter-helix interactions
	 # For chain length < 21, use subsegments of length 7; chains 22-27, len=9; 28-36, len=12

	## subsegment chain A
	#  Residue sub strings for vectors
	subseg_len_A 	= subsegment_len_dict[len_chA]
	midline 		= int( round( chain_A_CAs.getResnums()[-1] - chain_A_CAs.getResnums()[0] )/2 )

	N_term_selStrA 	= [ str( x ) for x in chain_A_CAs.getResnums()[ 0 : subseg_len_A ] ] 
	# finding the correct middle residue index to splice, then it should cover all residues, NOTE: this works for subseg_len_A = 7, confirmed so far
	M_term_selStrA 	= [ str( x ) for x in chain_A_CAs.getResnums()[ midline - int(round(subseg_len_A/2)) + 2 : midline + subseg_len_A - int(round(subseg_len_A/2) ) + 2 ] ] 
	C_term_selStrA	= [ str( x ) for x in chain_A_CAs.getResnums()[ - subseg_len_A : ] ] 

	N_term_resiA 	= chain_A.select('ca resnum %s' % ' '.join( N_term_selStrA) ) 
	M_term_resiA 	= chain_A.select('ca resnum %s' % ' '.join( M_term_selStrA) ) 
	C_term_resiA	= chain_A.select('ca resnum %s' % ' '.join( C_term_selStrA) ) 
	
	# move pdb such that the center of chain A is at the origin (0,0,0) 
	moveAtoms( chain_A_CAs, to=np.zeros(3), ag=True )
	
	# fit a helical axis vector to the middle subsection of chain A helix vector
	fit_result		= fit( M_term_resiA.getCoords() )
	projPoints_mA 	= ClosestPointOnLine( np.append( fit_result[0],fit_result[1]  ), chain_A_CAs.select('ca').getCoords() )

	# calculate rotation matrix to align vector of chain A middle segment to z axis, then apply this transformation
	rotation 		= VectorAlign( fit_result[0], np.array([0,0,1]) )
	transformZax	= Transformation( rotation , np.zeros(3) )
	applyTransformation( transformZax, pdb )
	writePDB('tmpAlign2.pdb', pdb)

	
	## calculate helix axis vector for each sub-segment of chain
	# 	project the CA atom onto it's respective helical axis vector, new coordinate for each CA 
	axes_A, cenz_A, points_on_Vects_A = [ [], [], [] ], [ [], [], [] ], [ [], [], [] ]  
	for n, coords in enumerate([N_term_resiA, M_term_resiA, C_term_resiA], 0):
		axes_A[n], cenz_A[n], null1, null2 	= fit( coords.getCoords() )
		points_on_Vects_A[n] 	= ClosestPointOnLine( np.append( axes_A[n], cenz_A[n] ), coords.select('ca').getCoords() )
		#write_markAtoms(points_on_Vects_A[n], './marker_A%d.pdb' % n)		## debugging for helix axis vectors


	#  use to then calculate projection of each residue onto helix axis
	#  this gives duplicate for some residues.... so use first instance of residue (N to C) to avoid duplication
	#print(N_term_selStrA, M_term_selStrA , C_term_selStrA)
	proj_coords_A, resid_2_coords_A, resid_2_vect_A = [], {}, {}
	for res_num in chain_A_CAs.getResnums():
	
		if str(res_num) in N_term_selStrA:
			index = N_term_selStrA.index( str(res_num) ) 
			proj_coords_A.append( points_on_Vects_A[0][index] )
			resid_2_vect_A[res_num] 	= 'N'	
			resid_2_coords_A[res_num] 	= points_on_Vects_A[0][index] 

		elif str(res_num) in C_term_selStrA:
			index = C_term_selStrA.index( str(res_num) ) 
			proj_coords_A.append( points_on_Vects_A[2][index] )
			resid_2_vect_A[res_num] 	= 'C'	
			resid_2_coords_A[res_num] 	= points_on_Vects_A[2][index] 	
		
		elif str(res_num) in M_term_selStrA:
			index = M_term_selStrA.index( str(res_num) ) 
			proj_coords_A.append( points_on_Vects_A[1][index] )
			resid_2_vect_A[res_num] 	= 'M'	
			resid_2_coords_A[res_num] 	= points_on_Vects_A[1][index] 

		else:
			print('\n\tResidue %d does not match known residues for chain %s: %s' % (res_num ,chains[0] , ' '.join([str(x) for x in chain_A_CAs.getResnums() ])) )
			sys.exit()

	#	print (res_num, resid_2_vect_A[res_num])

	## subsegment chain B
	#  Residue sub strings for vectors
	subseg_len_B 	= subsegment_len_dict[len_chB]
	midline 		= int( round( chain_B_CAs.getResnums()[-1] - chain_B_CAs.getResnums()[0] )/2 )

	N_term_selStrB 	= [ str( x ) for x in chain_B_CAs.getResnums()[ 0 : subseg_len_B ] ] 
	# finding the correct middle residue index to splice, then it should cover all residues, NOTE: this works for subseg_len_A = 7, confirmed so far
	M_term_selStrB 	= [ str( x ) for x in chain_B_CAs.getResnums()[ midline - int(round(subseg_len_B/2)) + 2 : midline + subseg_len_B - int(round(subseg_len_B/2) ) + 2 ] ] 
	C_term_selStrB	= [ str( x ) for x in chain_B_CAs.getResnums()[ - subseg_len_B : ] ] 

	N_term_resiB 	= chain_B.select('ca resnum %s' % ' '.join( N_term_selStrB) ) 
	M_term_resiB 	= chain_B.select('ca resnum %s' % ' '.join( M_term_selStrB) ) 
	C_term_resiB	= chain_B.select('ca resnum %s' % ' '.join( C_term_selStrB) ) 


	## calculate helix axis vector for each sub-segment of chain
	axes_B, cenz_B, points_on_Vects_B = [ [], [], [] ], [ [], [], [] ], [ [], [], [] ]  
	for n, coords in enumerate([N_term_resiB, M_term_resiB, C_term_resiB], 0):
		axes_B[n], cenz_B[n], null1, null2 	= fit( coords.getCoords() )
		points_on_Vects_B[n] 	= ClosestPointOnLine( np.append( axes_B[n], cenz_B[n] ), coords.select('ca').getCoords() )
		#write_markAtoms(points_on_Vects_B[n], './marker_B%d.pdb' % n)		## debugging for helix axis vectors


	#  use to then calculate projection of each residue onto helix axis
	#  this gives duplicate for some residues, use first instance (N to C)
	proj_coords_B, resid_2_coords_B, resid_2_vect_B = [], {}, {}
	#print(N_term_selStrB, M_term_selStrB , C_term_selStrB)
	for res_num in chain_B_CAs.getResnums():
	
		if str(res_num) in N_term_selStrB:
			index = N_term_selStrB.index( str(res_num) ) 
			proj_coords_B.append( points_on_Vects_B[0][index] )
			resid_2_vect_B[res_num] 	= 'N'	
			resid_2_coords_B[res_num] 	= points_on_Vects_B[0][index] 

		elif str(res_num) in C_term_selStrB:
			index = C_term_selStrB.index( str(res_num) ) 
			proj_coords_B.append( points_on_Vects_B[2][index] )
			resid_2_vect_B[res_num] 	= 'C'	
			resid_2_coords_B[res_num] 	= points_on_Vects_B[2][index] 

		elif str(res_num) in M_term_selStrB:
			index = M_term_selStrB.index( str(res_num) ) 
			proj_coords_B.append( points_on_Vects_B[1][index] )
			resid_2_vect_B[res_num] 	= 'M'	
			resid_2_coords_B[res_num] 	= points_on_Vects_B[1][index] 

		else:
			print('\n\tResidue %d does not match known residues for chain %s: %s' % (res_num ,chains[0] , ' '.join([str(x) for x in chain_B_CAs.getResnums() ])) )
			sys.exit()

		#print (res_num, resid_2_vect_B[res_num])

	# detect if chain B is parallel or anti-parallel vs chain A, based on the middle vector (A's N-term AA to mid-point, to midpoint, to B's N-term AA)
	midpointA, midpointB = int(subseg_len_A/2), int(subseg_len_B/2)
	crossing_angle = getDihedralV2( points_on_Vects_A[1][0], points_on_Vects_A[1][midpointA+1], points_on_Vects_B[1][midpointB+1], points_on_Vects_B[1][0] )
	parallel_FLAG = 1
	if np.fabs( crossing_angle ) > 90:
		parallel_FLAG = 0

	#subseg_pair_AP 	= {'N':'C', 'M':'M', 'C':'N'}
	#subseg_pair_P 	= {'N':'N', 'M':'M', 'C':'C'}
	subseg_pairing	= [{'N':'C', 'M':'M', 'C':'N'}, {'N':'N', 'M':'M', 'C':'C'}]

	#### Now, break up chain into all possible holes, initiate Hole class object; name by the residue of the i+3 position
	#		sort by N to C, assign to vector on hole's chain -- also assing interacting group 
	holes_chainA, holeA_IDs = [], []
	N_term_selStrA, M_term_selStrA , C_term_selStrA  = [int(x) for x in N_term_selStrA], [int(x) for x in M_term_selStrA], [int(x) for x in C_term_selStrA]
	
	# check n-termini for truncated holes, 3 residue partial holes: where r is i+3 (i+3, i+4, i+7) 
	for r in N_term_selStrA:	
		if r+4 in chain_A_CAs.getResnums() and r-3 not in chain_A_CAs.getResnums():
			#print (r, hole_sel_str, '--',len(hole.select('ca')) )
			proj_coords = [ resid_2_coords_A[ x ] for x in [r, r+1, r+4] ]
			holes_chainA.append( Hole( chain_A.select( 'resnum %d %d %d' % (r, r+1, r+4) ), chains[0] , [r, r+1, r+4], proj_coords, resid_2_vect_A[r] ) )


	# 	Look for 4 residue holes	;  find i, i+3, i+4, i+7
	for r in chain_A_CAs.getResnums():
		if r+7 <= chain_A_CAs.getResnums()[-1]:
			#hole_sel_str = 'resnum %d %d %d %d' % (r, r+3, r+4, r+7)
			#hole, holeID = chain_A.select( hole_sel_str ), r+3 
			#print (r, hole_sel_str, '-', len(hole.select('ca')) )
			proj_coords = [ resid_2_coords_A[ x ] for x in [r, r+3, r+4, r+7] ]
			holes_chainA.append( Hole( chain_A.select( 'resnum %d %d %d %d' % (r, r+3, r+4, r+7) ), chains[0] , [r, r+3, r+4, r+7], proj_coords, resid_2_vect_A[r+3]  ) )

	
	## check c-termini for truncated holes, 3 residue partial holes: where r is i+3 (i+3, i+4, i+7) 	
	for r in C_term_selStrA:	
		if r+3 in chain_A_CAs.getResnums() and r+4 in chain_A_CAs.getResnums() and r+7 not in chain_A_CAs.getResnums():
			#hole_sel_str = 'resnum %d %d %d' % (r, r+1, r+4)
			#hole = chain_A.select( hole_sel_str )
			#print (r, hole_sel_str, '--',len(hole.select('ca')) )
			proj_coords = [ resid_2_coords_A[ x ] for x in [r, r+3, r+4] ]
			holes_chainA.append( Hole( chain_A.select( 'resnum %d %d %d' % (r, r+3, r+4)), chains[0] , [r, r+3, r+4], proj_coords, resid_2_vect_A[r+3]  ) )



	holes_chainB, holeB_IDs = [], []
	N_term_selStrB, M_term_selStrB , C_term_selStrB  = [int(x) for x in N_term_selStrB], [int(x) for x in M_term_selStrB], [int(x) for x in C_term_selStrB]
	
	## check n-termini for truncated holes, 3 residue partial holes: where r is i+3 (i+3, i+4, i+7) 
	for r in N_term_selStrB:	
		if r+4 in chain_B_CAs.getResnums() and r-3 not in chain_B_CAs.getResnums():
			#hole_sel_str = 'resnum %d %d %d' % (r, r+1, r+4)
			#hole, holeID = chain_B.select( hole_sel_str ), r 
			#print (r, hole_sel_str, '--',len(hole.select('ca')) )
			proj_coords = [ resid_2_coords_B[ x ] for x in [r, r+1, r+4] ]
			holes_chainB.append( Hole( chain_B.select( 'resnum %d %d %d' % (r, r+1, r+4)), chains[1] , [r, r+1, r+4], proj_coords, resid_2_vect_B[r]    ) )

	## 	Look for 4 residue holes	;  find i, i+3, i+4, i+7
	for r in chain_B_CAs.getResnums():
		if r+7 <= chain_B_CAs.getResnums()[-1]:
			#hole_sel_str = 'resnum %d %d %d %d' % (r, r+3, r+4, r+7)
			#hole, holeID = chain_B.select( hole_sel_str ), r+3 
			#print (r, hole_sel_str, '-', len(hole.select('ca')) )
			proj_coords = [ resid_2_coords_B[ x ] for x in [r, r+3, r+4, r+7] ]
			holes_chainB.append( Hole( chain_B.select( 'resnum %d %d %d %d' % (r, r+3, r+4, r+7)), chains[1] , [r, r+3, r+4, r+7], proj_coords, resid_2_vect_B[r+3]  ) )
		
	for r in C_term_selStrB:	
		if r+3 in chain_B_CAs.getResnums() and r+4 in chain_B_CAs.getResnums() and r+7 not in chain_B_CAs.getResnums():
			#hole_sel_str = 'resnum %d %d %d' % (r, r+1, r+4)
			#hole, holeID = chain_B.select( hole_sel_str ), r 
			#print (r, hole_sel_str, '--',len(hole.select('ca')) )
			proj_coords = [ resid_2_coords_B[ x ] for x in [r, r+3, r+4] ]
			holes_chainB.append( Hole( chain_B.select( 'resnum %d %d %d' % (r, r+3, r+4) ), chains[1] , [r, r+3, r+4], proj_coords, resid_2_vect_B[r+3]  ) )


	# hash N, M, and C residue subsets of each chain for easy calling later
	B_resiList 	=	{ 'N':N_term_selStrB, 'M':M_term_selStrB, 'C':C_term_selStrB }
	A_resiList 	=	{ 'N':N_term_selStrA, 'M':M_term_selStrA, 'C':C_term_selStrA }
	B_resiSet 	=	{ 'N':N_term_resiB, 'M':M_term_resiB, 'C':C_term_resiB }
	A_resiSet 	=	{ 'N':N_term_resiA, 'M':M_term_resiA, 'C':C_term_resiA }
#	perCh_set 	= {chains[0]:A_resiSet, chains[1], B_resiSet}
#	hole_set 	= {chains[0]:holes_chainA, chains[1]:holes_chainB}


	
	#### For each hole, find possible interacting knobs
	## 		step 1: shorten list of potential interacting residues by subset pairing, N-N, C-C for par or N-C, C-C for antipar
	##		step 2: exclude residues with CA > 12 Angstrom away
	## 		step 3: vector method, exclude residues pointing away from hole's normal vector
	time_ = time.time()

########### chain A #################
	eligible_holesA = []
	for h in holes_chainA:
		h.print_hole_markers()

		print('\n---- ', h.fullID)
		# which vector do I interact with:  shortened residue list from opposing chain
		partner_resList = B_resiList[ subseg_pairing[parallel_FLAG][h.chain_subseg] ]
		partner_CAs 	= B_resiSet[ subseg_pairing[parallel_FLAG][h.chain_subseg] ]
		
		#check distance from 'hole' center to each potential knob CA.  skip any knobs > 12 Å away
		close_knobs, eligible_knobs = [], []
		for k, atom in zip(partner_resList,  partner_CAs.getCoords()) :
			#print (k, LA.norm( h.ca34cen  - atom))
			if LA.norm( h.ca34cen  - atom) < 12:
				close_knobs.append( k )

		### vector method; exclude knobs which point generally away from hole
		# knob unit vector - points from CA atom towards center of mass of sidechain, or for GLY: CA projection to CA
		# hole unit vector - cross product of CA 3-4 vector and CA's 1-7 vector (or 1-cen / cen-7 for 3aa holes) 
		# project these vector in 2D, z=0, z=y plane.  Calculate angle between vectors - exclude if > 120 degrees	
		#print('   >', close_knobs)
		for k in close_knobs:
			knob_id 	= '%s%d' % (chains[1],k)
			contacts 	= 0
			knob_aa 	= chain_B.select('resnum %s' % k)
			if 'GLY' in knob_aa.getResnames():
				knob_vect 		= knob_aa.select('ca').getCoords()[0] - resid_2_coords_B[ k ] 
				unit_knobVect	= knob_vect / LA.norm(knob_vect)
				#print (knob_vect, unit_knobVect, 'GLY')
			else:
				knob_sc			= knob_aa.select('not name CA C O N H')
				knob_vect 		= calcCenter( knob_sc, weights=knob_sc.getMasses()) - knob_aa.select('ca').getCoords()[0] 
				unit_knobVect	= knob_vect / LA.norm(knob_vect)
				#print(  unit_knobVect,  knob_aa.select('ca').getCoords()[0], calcCenter( knob_sc, weights=knob_sc.getMasses()) )
				#print( k,  unit_knobVect, 2* unit_knobVect +  calcCenter( knob_sc, weights=knob_sc.getMasses())) 
			
			# normal vector emergging from hole is roughly pointed towards knob sidechain vector
			# negtaive dot product means knob & hole pointing towards from each other, 
			interaction_vector 	=  np.dot( h.hole_vect[:2], unit_knobVect[:2] ) * 180/pi
			#print (k, round(interaction_vector))
			if interaction_vector > 10:	
				#print('eligible: ', k, round(interaction_vector, 1))	
				continue

			if h not in eligible_holesA:
				eligible_holesA.append(h)
				

			##### penultimate steps all-atom distance-based check of knobs #####

			# step 1) distance from hole marker atom.  
			#      knob w/ heavy atom < 3.1 Å is guranteed hole; >4.5 is ruled out as non-hole; else, all knob atoms check to hole bb atoms 
			closest_dist2Cen = np.min( buildDistMatrix( np.array( [h.hole_mark34 ]), knob_aa.select('not element H') ) )
			#if closest_dist2Cen < 3.1:
			if closest_dist2Cen < 4.5:
				h.knobs_by_contact[k] = Knob( chains[1], k, knob_aa )
				h.knobs_by_contact[k].dist_to_hole34cen = np.min(closest_dist2Cen) 
				print ( '\t-> potential knob %s \t near hole center: %.2f ' % (knob_id, closest_dist2Cen) )
			else:
				continue

			#  step 2) knobs contacting backbone atoms of holes: < 3.2 Å ; include all additional <=3.5 Å contacts
			total_contacts, hole_resi_reached, hole_atom_reached = 0, [], 0
			for r in h.resi_set:
				bb 			= h.atoms.select('resnum %s name N O C CA HA H 1HA 2HA' % r) 
				dMat 		= buildDistMatrix( bb , knob_aa)
				min_dist 	= np.min(dMat)
				if min_dist < 3.2:
					contacts 		   	= len( (dMat < 3.5).nonzero()[0] )
					hole_atom_reached 	+= len( set( (dMat < 3.5).nonzero()[0]) )
					total_contacts	  	+=  contacts
					hole_resi_reached.append(r)


			if total_contacts > 0:
				print ( '\t---> knob %s found:   ' % knob_id, '%d total close contacts to %d residues' % (total_contacts, len(hole_resi_reached)) )
				
				try:
					h.knobs_by_contact[k].total_contacts = total_contacts
					h.knobs_by_contact[k].hole_resi_reached = hole_resi_reached
					h.knobs_by_contact[k].hole_atom_reached = hole_atom_reached
				except IndexError:
					h.knobs_by_contact[k] = Knob( chains[1], k, knob_aa )
					h.knobs_by_contact[k].dist_to_hole34cen = np.min(closest_dist2Cen) 
					h.knobs_by_contact[k].total_contacts 	= total_contacts
					h.knobs_by_contact[k].hole_resi_reached = hole_resi_reached
					h.knobs_by_contact[k].hole_atom_reached = hole_atom_reached

				## FOR DEBUGGING:  write all knobs with close contacts
				writePDB( 'testKnob-%s_Hole_%s.pdb' % (knob_id, h.id ), h.atoms.copy() + h.knobs_by_contact[k].atoms.copy() ) 
				

			### Final SASA based check that knob burying some backbone atom solvent exposed area for both residues 3+4 
			#  this contributes to packing score, although distance alone is sufficent to be counted as a knob
			h.checkSASA( h.knobs_by_contact[k] )
				#for s in [h.knobs_by_contact[k].dSASA, h.knobs_by_contact[k].dSASA34, h.knobs_by_contact[k].dSASAcen]:
				#	print ( np.sum(s) )


			#### make evaluation of knob & its contribution to packing score
			# rule 1 close packing to the hole center <3.1 Å 
			if h.knobs_by_contact[k].dist_to_hole34cen < 3.1:
				h.official_Knobs[knob_id] = h.knobs_by_contact[k]
				packing_scoreTotal 	+= 1
				number_knobs		+= 1
				print ( '\t-----> official knob %s found   by rule 1 ' % knob_id, h.knobs_by_contact[k].dSASAcen )
				continue
			
			if round(closest_dist2Cen, 2) <=3.5:
				print(round(h.knobs_by_contact[k].dSASAcen,1), h.knobs_by_contact[k].dSASA34, h.knobs_by_contact[k].dSASA, len(hole_resi_reached), total_contacts)
				## partial knob, half point to packing score if dSASA_CEN >2 + 1 residue contacted
				if h.knobs_by_contact[k].dSASAcen > 5:
					h.partial_Knobs[knob_id] = h.knobs_by_contact[k]
					packing_scoreTotal 	+= 0.5
					print ( '\t-----> partial knob %s found  ' % knob_id )


			# rule 2 close packing to backbone <3.2 Å, AND 1 of the following
			if total_contacts > 1:
				print(round(h.knobs_by_contact[k].dSASAcen,1), h.knobs_by_contact[k].dSASA34, h.knobs_by_contact[k].dSASA, len(hole_resi_reached), total_contacts)
				#  (a) > 3 close contacts w/ >2 hole residues contacted
				if len(hole_resi_reached) >= 2 and total_contacts >= 3 :
					h.official_Knobs[knob_id] = h.knobs_by_contact[k]
					packing_scoreTotal 	+= 1
					number_knobs		+= 1
					print ( '\t-----> official knob %s found   by rule 2a ' % knob_id )
					continue
				#  (b) dSASA for both residue 3+4, and dSASA 3+4 is > 2
				if np.all(h.knobs_by_contact[k].dSASA34) and np.sum(h.knobs_by_contact[k].dSASA34) > 2:
					h.official_Knobs[knob_id] = h.knobs_by_contact[k]
					packing_scoreTotal 	+= 1
					number_knobs		+= 1
					print ( '\t-----> official knob %s found   by rule 2b ' % knob_id )
					continue
				
				## partial knob, half point to packing score if dSASA_CEN >2 + 1 residue contacted
				if h.knobs_by_contact[k].dSASAcen > 2:
					h.partial_Knobs[knob_id] = h.knobs_by_contact[k]
					packing_scoreTotal 	+= 0.5
					print ( '\t-----> partial knob %s found  ' % knob_id )
				
				

########### chain B #################
	eligible_holesB = []
	for h in holes_chainB:
		h.print_hole_markers()

		print('\n---- ', h.fullID)
		# which vector do I interact with:  shortened residue list from opposing chain
		partner_resList = A_resiList[ subseg_pairing[parallel_FLAG][h.chain_subseg] ]
		partner_CAs 	= A_resiSet[ subseg_pairing[parallel_FLAG][h.chain_subseg] ]
		
		#check distance from 'hole' center to each potential knob CA.  skip any knobs > 12 Å away
		close_knobs, eligible_knobs = [], []
		for k, atom in zip(partner_resList,  partner_CAs.getCoords()) :
			#print (k, LA.norm( h.ca34cen  - atom))
			if LA.norm( h.ca34cen  - atom) < 12:
				close_knobs.append( k )

		### vector method; exclude knobs which point generally away from hole
		# knob unit vector - points from CA atom towards center of mass of sidechain, or for GLY: CA projection to CA
		# hole unit vector - cross product of CA 3-4 vector and CA's 1-7 vector (or 1-cen / cen-7 for 3aa holes) 
		# project these vector in 2D, z=0, z=y plane.  Calculate angle between vectors - exclude if > 120 degrees	
		#print('   >', close_knobs)
		for k in close_knobs:
			knob_id 	= '%s%d' % (chains[0],k)
			contacts 	= 0
			knob_aa 	= chain_A.select('resnum %s' % k)
			if 'GLY' in knob_aa.getResnames():
				knob_vect 		= knob_aa.select('ca').getCoords()[0] - resid_2_coords_A[ k ] 
				unit_knobVect	= knob_vect / LA.norm(knob_vect)
				#print (knob_vect, unit_knobVect, 'GLY')
			else:
				knob_sc			= knob_aa.select('not name CA C O N H')
				knob_vect 		= calcCenter( knob_sc, weights=knob_sc.getMasses()) - knob_aa.select('ca').getCoords()[0] 
				unit_knobVect	= knob_vect / LA.norm(knob_vect)
				#print(  unit_knobVect,  knob_aa.select('ca').getCoords()[0], calcCenter( knob_sc, weights=knob_sc.getMasses()) )
				#print( k,  unit_knobVect, 2* unit_knobVect +  calcCenter( knob_sc, weights=knob_sc.getMasses())) 
			
			# normal vector emergging from hole is roughly pointed towards knob sidechain vector
			# negtaive dot product means knob & hole pointing towards from each other, 
			interaction_vector 	=  np.dot( h.hole_vect[:2], unit_knobVect[:2] ) * 180/pi
			#print (k, round(interaction_vector))
			if interaction_vector > 10:	
				#print('eligible: ', k, round(interaction_vector, 1))	
				continue

			if h not in eligible_holesB:
				eligible_holesB.append(h)
				

			##### penultimate steps all-atom distance-based check of knobs #####

			# step 1) distance from hole marker atom.  
			#      knob w/ heavy atom < 3.1 Å is guranteed hole; >4.5 is ruled out as non-hole; else, all knob atoms check to hole bb atoms 
			closest_dist2Cen = np.min( buildDistMatrix( np.array( [h.hole_mark34 ]), knob_aa.select('not element H') ) )
			#if closest_dist2Cen < 3.1:
			if closest_dist2Cen < 4.5:
				h.knobs_by_contact[k] = Knob( chains[0], k, knob_aa )
				h.knobs_by_contact[k].dist_to_hole34cen = np.min(closest_dist2Cen) 
				print ( '\t-> potential knob %s \t near hole center: %.2f ' % (knob_id, closest_dist2Cen) )
			else:
				continue

			#  step 2) knobs contacting backbone atoms of holes: < 3.2 Å ; include all additional <=3.5 Å contacts
			total_contacts, hole_resi_reached, hole_atom_reached = 0, [], 0
			for r in h.resi_set:
				bb 			= h.atoms.select('resnum %s name N O C CA HA H 1HA 2HA' % r) 
				dMat 		= buildDistMatrix( bb , knob_aa)
				min_dist 	= np.min(dMat)
				if min_dist < 3.2:
					contacts 		   	= len( (dMat < 3.5).nonzero()[0] )
					hole_atom_reached 	+= len( set( (dMat < 3.5).nonzero()[0]) )
					total_contacts	  	+=  contacts
					hole_resi_reached.append(r)


			if total_contacts > 0:
				print ( '\t---> knob %s found:   ' % knob_id, '%d total close contacts to %d residues' % (total_contacts, len(hole_resi_reached)) )
				
				try:
					h.knobs_by_contact[k].total_contacts = total_contacts
					h.knobs_by_contact[k].hole_resi_reached = hole_resi_reached
					h.knobs_by_contact[k].hole_atom_reached = hole_atom_reached
				except IndexError:
					h.knobs_by_contact[k] = Knob( chains[1], k, knob_aa )
					h.knobs_by_contact[k].dist_to_hole34cen = np.min(closest_dist2Cen) 
					h.knobs_by_contact[k].total_contacts 	= total_contacts
					h.knobs_by_contact[k].hole_resi_reached = hole_resi_reached
					h.knobs_by_contact[k].hole_atom_reached = hole_atom_reached

				## FOR DEBUGGING:  write all knobs with close contacts
				writePDB( 'testKnob-%s_Hole_%s.pdb' % (knob_id, h.id ), h.atoms.copy() + h.knobs_by_contact[k].atoms.copy() ) 
				

			### Final SASA based check that knob burying some backbone atom solvent exposed area for both residues 3+4 
				#  this contributes to packing score, although distance alone is sufficent to be counted as a knob
			h.checkSASA( h.knobs_by_contact[k] )
				#for s in [h.knobs_by_contact[k].dSASA, h.knobs_by_contact[k].dSASA34, h.knobs_by_contact[k].dSASAcen]:
				#	print ( np.sum(s) )


			#### make evaluation of knob & its contribution to packing score
			# rule 1 close packing to the hole center <3.1 Å 
			if h.knobs_by_contact[k].dist_to_hole34cen < 3.1:
				h.official_Knobs[knob_id] = h.knobs_by_contact[k]
				packing_scoreTotal 	+= 1
				number_knobs		+= 1
				print ( '\t-----> official knob %s found   by rule 1 ' % knob_id, h.knobs_by_contact[k].dSASAcen )
				continue
			
			if round(closest_dist2Cen, 2) <=3.5:
				print(round(h.knobs_by_contact[k].dSASAcen,1), h.knobs_by_contact[k].dSASA34, h.knobs_by_contact[k].dSASA, len(hole_resi_reached), total_contacts)

			# rule 2 close packing to backbone <3.2 Å, AND 1 of the following
			if total_contacts > 1:
				print(round(h.knobs_by_contact[k].dSASAcen,1), h.knobs_by_contact[k].dSASA34, h.knobs_by_contact[k].dSASA, len(hole_resi_reached), total_contacts)
				#  (a) > 3 close contacts w/ >2 hole residues contacted
				if len(hole_resi_reached) >= 2 and total_contacts >= 3 :
					h.official_Knobs[knob_id] = h.knobs_by_contact[k]
					packing_scoreTotal 	+= 1
					number_knobs		+= 1
					print ( '\t-----> official knob %s found   by rule 2a ' % knob_id )
					continue
				#  (b) dSASA for both residue 3+4, and dSASA 3+4 is > 2
				if np.all(h.knobs_by_contact[k].dSASA34) and np.sum(h.knobs_by_contact[k].dSASA34) > 2:
					h.official_Knobs[knob_id] = h.knobs_by_contact[k]
					packing_scoreTotal 	+= 1
					number_knobs		+= 1
					print ( '\t-----> official knob %s found   by rule 2b ' % knob_id )
					continue
				
				## partial knob, half point to packing score if dSASA_CEN >2 and total sSASA > 4 ; 1 residue contacted
				if h.knobs_by_contact[k].dSASAcen > 2:
					h.partial_Knobs[knob_id] = h.knobs_by_contact[k]
					packing_scoreTotal 	+= 0.5
					print ( '\t-----> partial knob %s found  ' % knob_id )

	

			

		#h.delete_hole_marker_file()
					
			
	print('\n')
	print ( round( time.time() - time_, 6), 'time elapsed (s)'  )

	return	(number_knobs, packing_scoreTotal)
		

		





























