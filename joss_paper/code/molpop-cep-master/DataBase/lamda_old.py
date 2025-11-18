#/usr/bin/env python
from __future__ import print_function
try:
	from html.parser import HTMLParser
except:
	from HTMLParser import HTMLParser

try:
	from urllib.request import urlopen
except:
	from urllib import urlopen
	

class AnchorParser(HTMLParser):
    "Basic HTML parser that gathers a set of all href values in a webpage by targetting the anchor tag"

    def __init__(self, baseURL = ""):
        """Constructor for AnchorParser
        Args:
            baseURL (str): Base URL for the HTML content
        Returns:
            None
        """
        # Parent class constructor
        HTMLParser.__init__(self)
        # Set of all hyperlinks in the web page
        self.pageLinks = set()
        # The base url of the webpage to parse
        self.baseURL = baseURL

    def getLinks(self):
        """Return the set of absolute URLs in the HTML content
        Returns:
            set: Absolute URLs found in HTML content
        """
        return self.pageLinks

    def handle_starttag(self, tag, attrs):
        """Override handle_starttag to target anchor tags
        Returns:
            None
        """
        # Identify anchor tags
        if tag == "a":
            for(attribute, value) in attrs:
                # Anchor tags may have more than 1 attribute, but handle_starttag will only target href
                # Attribute examples: href, target, rel, etc
                # Attribute list can be found at: https://www.w3schools.com/tags/tag_a.asp
                if attribute == "href":
                    # Form an absolute URL based on the relative URL
                    absoluteUrl = urljoin(self.baseURL, value)
                    # We want to avoid href values that are not http/https
                    # Example: <a href="mailto:person@some.com">Send Email Now!</a>
                    if urlparse(absoluteUrl).scheme in ["http", "https"]:
                        # Once a new hyperlink is obtained, add it to the set
                        self.pageLinks.add(absoluteUrl)

class AnchorParser2(HTMLParser):
	def __init__(self, verbose, datafiles):
		HTMLParser.__init__(self)		
		self.loop = 0
		self.files = []
		self.goodSection = False
		self.inside = False
		self.recording = False
		self.data = []
		self.verbose = verbose
		self.datafiles = datafiles		
		
	def setLimits(self, labelFrom, labelTo):
		self.labelFrom = labelFrom
		self.labelTo = labelTo
		
	def handle_starttag(self, tag, attrs):
		if (self.datafiles):
			if (tag.lower() =='a'):
				for key, value in attrs:
					if (key.lower() == 'href'):						
						if (value.find("http://www.strw.leidenuniv.nl/~moldata/datafiles/") != -1):
							if (self.verbose):
								print(self.loop, value)
							self.loop = self.loop + 1
							self.files.append(value)
		else:
			if (tag.lower() == 'div' and attrs[0][1] == 'sidebar'):
				self.inside = True
				
			if (tag.lower() == 'b'):
				self.recording = True
				
			if (tag.lower() =='a' and self.inside and self.goodSection):
				for key, value in attrs:
					if (key.lower() == 'href'):						
						if (value.find("http://www.strw.leidenuniv.nl/~moldata/") != -1):
							if (self.verbose):
								print(self.loop, value)
							self.loop = self.loop + 1
							self.files.append(value)
						
	def handle_endtag(self, tag):
		if (tag.lower() == 'div' and self.inside):
			self.inside = False
			
	def handle_data(self, data):
		if (self.recording):
			if (data.find(self.labelFrom) != -1):
				self.goodSection = True				
			if (data.find(self.labelTo) != -1):
				self.goodSection = False				
				
						
class molpop():
	def __init__(self):
		pass
	def parse(self, file, data):
		
# File with energy levels and Einstein A coefficients
		print("Writing radiative data...")
		
		molName = file.split('/')[-1].split('.')[0]
		nLevels = int(data[5])
		molMass = float(data[3])

		name_mol = data[1].split()[0]
		
		f = open(molName+"_lamda.molecule", "w")
		f.write('MOLPOP Energy levels and A-coefficients file\n')
		f.write('Generated from the Leiden database file '+molName+'\n\n')
		f.write('Molecular species\n')
		f.write('>\n')
		f.write(name_mol+'\n')		
		f.write('\n')
		f.write('N. levels and molecular mass\n')
		f.write('>\n')
		f.write('%d   %f\n'%(nLevels,molMass))
		f.write('\n')
		f.write('   N     g      Energy in cm^{-1}     Level details\n')
		f.write('>\n')
		
		for i in range(nLevels):
			temp = data[7+i].split()
			f.write("{0:5d}   {1:3d}  {2:18.6f}        '{3}'\n".format(int(temp[0]),int(float(temp[2])),float(temp[1]),temp[3]))
			
		f.write('\n')
		f.write('Einstein coefficients A_ij\n')
		f.write('Reference: LAMDA\n')
		f.write('\n')
		f.write('   i     j    A_ij in s^{-1}\n')
		f.write('>\n')
				
		
		nTransitions = int(data[7+nLevels+1])
		for i in range(nTransitions):
			temp = data[7+nLevels+3+i].split()
			f.write("{0:5d}   {1:3d}  {2:12.3e}\n".format(int(float(temp[1])),int(float(temp[2])),float(temp[3])))
			
		f.close()
						
		pointer = 7 + nLevels + 1 + nTransitions + 1 + 1 + 1

# File with collisional rates
		nCollPartners = int(data[pointer])
		pointer += 1
		
		print("Writing collisional data...")
		for i in range(nCollPartners):
			pointer += 1
			
			collisionDescription = data[pointer]
			whichCollision = data[pointer].replace(" - ","-").split()
			
			temp = whichCollision[1].split('-')
			
			fileName = molName+'_'+temp[1]+"_lamda.kij"
			
			print("Collisions with "+temp[1]+" -> "+'Coll/'+fileName)
			
			f = open('Coll/'+fileName, "w")
			
			pointer += 2
			nCollisions = int(data[pointer])
			
			pointer += 2
			nTemperatures = int(data[pointer])
			
			pointer += 2
			temp = data[pointer].split()
			Temperatures = list(map(float, temp))
			
			pointer += 2
		
			f.write('MOLPOP collision rates generated from the Leiden database file {0}\n'.format(molName))
			f.write(collisionDescription[2:])
			f.write('\n')
			f.write('>\n')
			f.write('\n')
			f.write('Number of temperature columns = '+str(nTemperatures)+'\n')
			f.write('\n')
			f.write('I    J                        TEMPERATURE (K)\n')
			f.write('\n')
			f.write(("          "+"{:12.2f}"*len(Temperatures)+"\n").format(*Temperatures))
			f.write('\n')
			
			for j in range(nCollisions):
				temp = data[pointer].split()
				coll = list(map(float, temp))
				f.write((("{:5d}   {:3d}  "+("{:12.3e}")*len(coll[3:])+"\n").format(int(coll[1]), int(coll[2]), *coll[3:])))
				pointer += 1
				
			f.close()
			

# Parse the directory with all the molecules
parser = AnchorParser('http://home.strw.leidenuniv.nl/~moldata/datafiles') #False, False)
# parser.setLimits('Molecular datafiles', 'Radiative transfer')
data = urlopen('http://home.strw.leidenuniv.nl/~moldata/').read().decode('utf-8', 'replace')
print("List of available molecules")
parser.feed(data)

breakpoint()


parser2 = AnchorParser(True, True)
parser2.setLimits('Datafiles', 'Links')
for f in parser.files:
	data = urlopen(f).read().decode('utf-8', 'replace')
	parser2.feed(data)

# Select the desired molecule
try:
	nb = input("Select which one to download (separated with spaces if many) (0-"+str(parser2.loop-1)+") ")
except:
	nb = raw_input("Select which one to download (separated with spaces if many) (0-"+str(parser2.loop-1)+") ")

nb = nb.split()

for iterator in nb:
	indexMolecule = int(iterator)
	print("Downloading "+parser2.files[indexMolecule])

# Download the molecule and parse it, generating the appropriate files
	ur = urlopen(parser2.files[indexMolecule])
	data = ur.readlines()

	for i in range(len(data)):
		data[i] = data[i].decode('utf-8', 'replace')

	mol = molpop()

	mol.parse(parser2.files[indexMolecule],data)
