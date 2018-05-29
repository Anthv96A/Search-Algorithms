import random
import math
import collections
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

#Use to import KMeans library

#nltk.download('stopwords')
#nltk.download('punkt')


def relocateGoal():
	
	rand = random.randint(1, coordinatesLength);
	if(rand > 90):
		## Opponent fights back
		newGoalPosition = random.randint(1, coordinatesLength);
		global goal
		if(goal in coords[newGoalPosition][0]):
			print("Goal not changed. Recursive call")
			relocateGoal();
		else:
			goal = coords[newGoalPosition][0]
			print("Target has now relocated to ", goal )


def extractSentences():
	sentences = [];
	filtered = [];
	
	for t in route:
		if t not in filtered:
			filtered.append(t)
	
	for i in range(len(filtered)):
		for j in range(len(coords)):
			if(filtered[i] in coords[j][0] ):
				sentences.append(coords[j][3]);
	return sentences;
				


def tokenizer(text):
	tokens = word_tokenize(text)
	stemmer = PorterStemmer() # Find words in words 
	tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
	return tokens

def KMeansClusterSentences(sentences, num_of_clusters=10):
	tfidf_vectorizer = TfidfVectorizer(tokenizer = tokenizer, stop_words = stopwords.words('english'), lowercase = True)
	tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
	kmeans = KMeans(n_clusters = num_of_clusters)
	kmeans.fit(tfidf_matrix)
	clusters = collections.defaultdict(list)
	for i, label in enumerate(kmeans.labels_):
		clusters[label].append(i)
	return dict(clusters)
	


# Breadth First Search
def bfs(x):
	c = 0
	neighbours = []
	for c in range(len(world)): # for all pairs [f,t] stop at the first pair [x,t]
		f = world[c][0]
		t = world[c][1]
		if(x == f): 
			neighbours.append(t);
			
	return neighbours; # Returns a list of neighbours from node
	
# Depth First Search	
def dfs(x):
	next = 'z'
	c = 0
	for c in range(len(world)):
		if(x == world[c][0]):
			next = world[c][1];
			break;
			
	return next;

	
	
def euclidean(x1, y1, x2, y2):
		xs = x1 + x2
		ys = y1 + y2
		return math.sqrt(pow(xs,2) + pow(ys,2)) 
		
	  
def manhattan(current_cell_x, current_cell_y, goal_x, goal_y):
		xs,ys = current_cell_x,current_cell_y
		xe,ye = goal_x,goal_y
		return abs(xs - xe) + abs(ys - ye);
		
		
def diagonal(current_cell_x, current_cell_y, goal_x, goal_y):
		xs,ys = current_cell_x, current_cell_y
		xe,ye = goal_x, goal_y
		return abs(abs(xs - xe) - abs(ys - ye));
		

  
		
def idastar(pathTaken):

	goalX = 0;
	goalY = 0;
	
	# Extract goal coords
	for i in range(len(coords)):
		if(goal in coords[i][0]):
			goalX = coords[i][1]
			goalY = coords[i][2]
			break;
	

	
	for nodes in range(len(pathTaken)):
		currentDist = 0;
		previousDist = 0;
		
		next = pathTaken[nodes]
		if(nodes -1 != -1):
			previous = pathTaken[nodes - 1]
		
		
		for i in range(len(coords)):
			if(next in coords[i][0]):
	
				currentX = coords[i][1] # Starting X coordinate
				currentY = coords[i][2] # Starting Y coordinate
				currentDist = euclidean( currentX, currentY, goalX, goalY )
				#print("Current distance ", currentDist);
					
				if(nodes -1 != -1):
					# Get previous euclidean distance
					previousX = coords[i - 1][1]
					previousY = coords[i - 1][2]
					previousDist = euclidean( previousX, previousY, goalX, goalY )
					#print("Previous distance ", previousDist);
			
					tempCurrentDist = currentDist
					tempPreviousDist = previousDist
					
					if(tempCurrentDist < tempPreviousDist):
						#print("CURRENT node is a shorter distance", next, "with a distance of ", tempCurrentDist)
						ida.append(pathTaken[nodes])
					else:
						#print("PREVIOUS node is a shorter distance", previous, "with a distance of ", tempPreviousDist);
						ida.append(pathTaken[nodes -1])
					
					if(next == goal):
						print("IDA found goal", next)
						break
									
					currentDist = 0
					previousDist = 0
									
	print("Iterative Deepening A* Route ",ida)
	return ida;
		



coords  = [ 
		['a',1,12, "Writing a list of random sentences is harder than I initially thought it would be."],
		['b',2,12, "I really want to go to work, but I am too sick to drive."],
		['c',3,12, "The mysterious diary records the voice."],
		['d',4,12, "How was the math test?"],
		['e',5,12, "Mary plays the piano."], 
		['f',6,12, "Artificial Intelligence is a growing topic these days"], 
		#['g',7,12, "Investing in the stock market and trading with them are not that easy"], 
		['h',8,12, "We have never been to Asia, nor have we visited Africa."],
		#['i',9,12, "We have been to Europe, but not Asia"], 
		['j',10,12,"She borrowed the book from him many years ago and hasn't yet returned it."],
		['k',11,12,"The body may perhaps compensates for the loss of a true metaphysics."],
		['l',12,12, "Quantum physics is ridiculously harder than most subjects"], 
		['m',1,11, "Don't step on the broken glass."],
		['n',2,11,"He told us a very exciting adventure story."],
		['o',3,11, "Where do random thoughts come from?"],
		['p',4,11, "Check back tomorrow; I will see if the book has arrived."],
		['q',5,11, "The memory we used to share is no longer coherent."], 
		['r',6,11, "A purple pig and a green donkey flew a kite in the middle of the night and ended up sunburnt."],
		['s',7,11, "Let me help you with your baggage."], 
		['t',8,11, "The quick brown fox jumps over the lazy dog."],
		['u',9,11, "He said he was not there yesterday; however, many people saw him there."],
		['v',10,11, "Wednesday is hump day."],
		#['w',11,11, "She wrote him a long letter, but he didn't read it."],
		#['x',12,11, "I am never at home on Sundays."],
		['y',1,10, "Wow, does that work?"],
		['z',2,10, "Sometimes it is better to just walk away from things"],
		['A',3,10, "There was no ice cream in the freezer, nor did they have money to go to the store."],
		['B',4,10, "Tom got a small piece of pie."],
		['C',5,10, "He turned in the research paper on Friday; otherwise, he would have not passed the class."],
		['D',6,10, "He ran out of money, so he had to stop playing poker."],
		['E',7,10, "She always speaks to him in a loud voice."], 
		['F',8,10, "The shooter says goodbye to his love."],
		#['G',9,10, "I think I will buy the red car, or I will lease the blue one."],
		['H',10,10, "Malls are great places to shop; I can find everything I need under one roof."],
		['I',11,10, "She was too short to see over the fence."],
		['J',12,10, "We need to rent a room for our party."],
		#['K',1,9, "I currently have 4 windows open up."],
		#['L',2,9, "The old apple revels in its authority."],
		#['M',3,9, "My attendance was not good enough."],
		#['N',4,9, "The lake is a long way from here."], 
		#['O',5,9, "She did not cheat on the test, for it was not the right thing to do."],
		#['P',6,9, "Please wait outside of the house."],
		#['Q',7,9, "It was getting dark, and we were not there yet."],
		#['R',8,9, "I am happy to take your donation, any amount will be greatly appreciated."], 
		#['S',9,9, "Warren Buffett is a genius in the stock market, hence why he is rich"],
		#['T',10,9, "If I do not like something, I will stay away from it."],
		#['U',11,9, "She advised him to come back at once."], 
		['V',12,9, "She did her best to help him."],
		['W',1,8, "Rock music approaches at high velocity."],
		['X',2,8, "I want more detailed information."],
		#['Y',3,8, "This is the last random sentence I will be writing and I am going to stop mid-sent"],
		['Z',4,8, "The book is in front of the table."],
		['1',5,8, "The clock within this blog and the clock on my laptop are 1 hour different from each other."],
		['2',6,8, "I am counting my calories, yet I really want dessert."], 
		['3',7,8, "Lets all be unique together until we realise we are all the same."], 
		['4',8,8, "The waves were crashing on the shore, it was a lovely sight."], 
		#['5',9,8, "Cats are good pets, for they are clean and are not noisy."],
		#['6',10,8, "My Mum tries to be cool by saying that she likes all the same things that I do."],
		['7',11,8, "I love eating toasted cheese and tuna sandwiches."], 
		['8',12,8, "Is it free?"],
		['9',1,7, "Two seats were vacant."],
		['10',2,7, "The sky is clear, the stars are twinkling."],
		#['11',3,7, "This is a Japanese doll."],
		['12',4,7, "Everyone was busy, so I went to the movie alone."], 
		#['13',5,7, "She only paints with bold colors; she does not like pastels."],
		['14',6,7, "We have a lot of rain in June."], 
		#['15',7,7, "Abstraction is often one floor above you."], 
		['16',8,7, "Yeah, I think it's a good environment for learning English."],
		['17',9,7, "They got there early, and they got really good seats."],
		#['18',10,7, "Italy is my favorite country, in fact, I plan to spend two weeks there next year."], 
		#['19',11,7, "I checked to make sure that he was still alive."], 
		['20',12,7, "A glittering gem is not enough."],
		['21',1,6, "A song can make or ruin a persons day if they let it get to them."],
		['22',2,6, "When I was little I had a car door slammed shut on my hand. I still remember it quite vividly."],
		#['23',3,6, "Christmas is coming."], 
		['24',4,6, "He did not want to go to the dentist, yet he went anyway."],
		#['25',5,6, "Should we start class now, or should we wait for everyone to get here?"],
		['26',6,6, "The stranger officiates the meal."],
		#['27',7,6, "The quick brown fox jumps over the lazy dog."],
		['28',8,6, "It was getting dark, and we were not there yet."],
		#['29',9,6, "Two seats were vacant."],
		['30',10,6, "I hear that Nancy is very pretty."],
		['31',11,6, "Dogs are more fun than cats"],
		['32',12,6, "The river stole the gods."], 
		['33',1,5, "Tom got a small piece of pie."],
		['34',2,5, "Please wait outside of the house."], 
		#['35',3,5, "The lake is a long way from here."],
		['36',4,5, "The feminine commercial can't moor the escape."],
		#['37',5,5, "The distorted pop can't tick the rain."],
		['38',6,5, "Did the brief bike really pump the delivery?"],
		#['39',7,5, "The dysfunctional dress can't mourn the listen."], 
		['40',8,5, "The perfect lack borrows into the embarrassed command."], 
		#['41',9,5, "The neat perception follows into the flagrant fear."], 
		['42',10,5, "It was then the subtle slip met the blue box."],
		#['43',11,5, "Did the melodic transition really steer the monitor?"], 
		#['44',12,5, "The rightful valuable can't joke the inspection."],
		['45',1,4, "Is the trust reflection better than the reserve?"],
		#['46',2,4, "South east Asia is my favourite place!"], 
		#['47',3,4, "The sour construction can't squash the rush."],
		['48',4,4, "Bodybuilding is fun"],
		#['49',5,4, "Cardio is boring"],
		['50',6,4, "Eating bad food is good for the soul"],
		#['51',7,4, "Good foods are good for you"],
		['52',8,4, "The full statement works into the ragged primary."],
		['53',9,4, "The rustic chest can't head the topic."],
		['54',10,4, "Is the coach level better than the ability?"],
		['55',12,4, "The premium pair impress into the straight few."],
		['56',12,4, "First class honours or nothing"],
		['57',1,3, "Gym is where bodybuilding is done"],
		['58',2,3, "It was then the coordinated target met the new mention."], 
		#['59',3,3, "Is the change web better than the dish?"], 
		['60',4,3, "Is the peep fly better than the access?"],
		#['61',5,3, "I hate the green flashing light."],
		['62',6,3, "DO NOT DISTURB, evil genius at work."],
		#['63',7,3, " Love your enemies, it makes them angry."],
		['64',8,3, "Did my sarcasm hurt your feels? Get over it."],
		#['65',9,3, "But my tree only hit the car in self-defence!"],
		#['66',10,3, "It is much funnier now that I get it."],
		#['67',11,3, "Come to the dark side. We have cookies."],
		['68',12,3, "I am not weird, I am gifted."],
		#['69',1,2, "Never set yourself on fire."],
		['70',2,2, "The banana has legs!"],
		#['71',3,2, "Do not worry, I was born this way."],
		['72',4,2, "I am not random! I just have lots of thoughts."],
		#['73',5,2, "Back off! The ice cream is mine!"],
		['74',6,2, "Tomorrow has been cancelled due to lack of interest."],
		#['75',7,2, "Even my issues have issues."],
		['76',8,2, "Angry people need hugs or sharp objects"], 
		#['77',9,2, "I here voices and they do not like you!"],
		['78',10,2, "I like eggs."],
		['79',11,2, "I am here to install your cushion."],
		['80',12,2, "Your hands are really hairy."],
		#['81',1,1, "I can think!"],
		['82',2,1, "We are so skilled!"],
		['83',3,1, "I am thinking bananas"],
		['84',4,1, "I have a magical box and it is better than yours."],
		#['85',5,1, "Caution! There is water on the road during rain."], 
		['86',6,1, "I am here to install your cushion."],
		#['87',7,1, "I do whatever my Rice Crispies tell me to do"],
		['88',8,1, "You sound like yourself"], 
		#['89',9,1, "Brusselsprouts are green!"], 
		['90',10,1, "Bananas can be green"], 
		['91',11,1, "I know kung fu and 50 other dangerous words."], 
		['92',12,1, "Finally, the last sentence!!!!"]
	]
	
	
world = [
	['a','b'], ['b','c'], ['c','d'], ['d','e'], ['e','f'], #First row
	['f','r'], # Directly moves down from node 'f'
	['r','q'], ['q','p'], ['p','o'], ['o','n'], ['n','m'], # Moving down the tree
	['m','y'], ['y','K'], ['K','W'] ,['y','z'], ['z','A'], ['A','B'], ['B','C'], ['C','D'],
	['W','X'], ['X','10'],
	['10','21'], ['10','22'], ['22','33'], ['22','34'], 
	['33','45'], ['45','57'], ['57','58'], ['58','70'],
	['70','82'], ['82','83'], ['83','84'],
	['84','72'], ['72','60'], ['60','48'], ['48', '36'], ['36','24'], ['24','12'], ['12','Z'],
	['Z','1'], ['1','2'], ['2','3'], ['3','4'], ['4','16'], ['16','28'], ['28','40'], ['40','52'], ['52','64'], ['64','76'], ['76','88'],
	['2','14'], ['14','26'], ['26', '38'], ['38','50'], ['50','62'], ['62','74'], ['74','86'],
	['52','53'], ['53','54'], ['54','55'], ['55','56'], ['56','68'], ['68','80'],
	['80','79'], ['80','78'], ['80','92'], ['80','91'], ['80','90'],
	['54','42'], ['42','30'], ['30','31'], ['31','32'], ['32','20'], ['20','8'],
	['8','7'], ['8','V'], ['V','J'], ['J','I'], ['I','H'], ['H','v'], ['v','j'], ['j','k'], ['k','l'],
	['v','u'], ['u','t'], ['t','F'], ['t','h']
	
	# Mirror World by flipping each node
]


#world = [ ['a','b'], ['a','c'], ['a','r'], ['r', 'x'], ['r', 'd'], ['x','y']]
	
initial = 'a' # Initial starting point
goal = 'h' # END GOAL SHOULD BE 
route = []
queue = []
queue.append(initial) # Put initial location in the queue
route.append(initial)
coordinatesLength = len(coords);
print("Robot at ", initial," goal ", goal)
found = False

euclideanDist = 0
manhattanDist = 0
diagonalDist = 0

# This list will be the values of the best route
# When the search is complete, compare the 'route' for each path taken
ida = []

while queue != None: # If the list is empty, then stop
	
	

	### Loop through the head of the queue for finding adjacent neighbours
	for q in range(len(queue)):
	
		tail = queue[1:len(queue)] # Get the remainder of the queue 
	
		newQueue = []
		
		for i in range(len(queue)):	# Get the adjacent neighbours
			if(found):
				break;
			else:
			 	relocateGoal()
			
				subList = bfs(queue[i]) # Get the adjacent neighbours from the queue
				newQueue += tail
				newQueue += subList
				
				
				for k in range(len(newQueue)):
					if(newQueue[k] not in queue):
						queue.append(newQueue[k])

				
				for j in range(len(subList)):	
					route.append(subList[j])
					
					if(subList[j] == goal):	# If node equals goal
						print("ROBOT FOUND GOAL" , goal) # Found the target goal
						extractVal = 'start' # 'start' for dummy data
						found = True #Break away when goal has been found
						for k in range(len(coords)): 
							if( initial in coords[k][0] ):
								extractVal = coords[k] # Extract the initial position 
						
							if( goal in coords[k][0]): # Find goal coordinates in 'coords'
								Xs = extractVal[1] # Starting X coordinate
								Ys = extractVal[2] # Starting Y coordinate
								Xe = coords[k][1] # Ending X coordinate
								Ye = coords[k][2] # Ending Y coordinate
									
								euclideanDist = euclidean( Xs, Ys, Xe, Ye ) # Using initial position coords and target coordinates to calculate euclidean distance
								manhattanDist = manhattan( Xs, Ys, Xe, Ye )
								diagonalDist = diagonal( Xs, Ys, Xe, Ye )
								
								break
					
						break
				

	if(found):
		print("Got to goal, calculated Euclidean distance of " , euclideanDist)
		print("Got to goal, calculated Manhattan distance of " , manhattanDist)
		print("Got to goal, calculated Diagonal distance of " , diagonalDist)
		
		avg = (euclideanDist + manhattanDist + diagonalDist ) / 3
		
		print("The average distance ", avg)
		idastar(route)
		
		extractedVals = extractSentences();
		
		nclusters = 10
		clusters = KMeansClusterSentences(extractedVals, nclusters);
		
		for cluster in range(nclusters):
			print("CLUSTER", cluster + 1, ":")
			for i, sentence in enumerate(clusters[cluster]):
				print("SENTENCE ", i + 1, ": ", extractedVals[sentence])
		
		break	
	

		
		

		