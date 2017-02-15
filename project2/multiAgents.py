# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    if currentGameState.isLose(): 
      return -float("inf")
    elif currentGameState.isWin():
      return float("inf")
    foodlist = currentGameState.getFood().asList()
    closestDistancefood=float("inf")
    for food in foodlist :
      temp_distance=util.manhattanDistance(newPos,food)
      closestDistancefood=min(temp_distance,closestDistancefood)
    scaredGhosts, activeGhosts = [], []
    for ghost in newGhostStates:
      if ghost.scaredTimer:
        scaredGhosts.append(ghost)
      else:
        activeGhosts.append(ghost)



    closestActiveGDistance=float("inf")
    if activeGhosts:
      for ghost in activeGhosts:
        temp_distance=util.manhattanDistance(newPos,ghost.getPosition())
        closestActiveGDistance=min(temp_distance,closestActiveGDistance)
    else:
      closestActiveGDistance=float("inf")


    closestActiveGDistance = max(closestActiveGDistance, 5)
    if scaredGhosts:
      for ghost in scaredGhosts:
        temp_distance = cutil.manhattanDistance(newPos,ghost.getPosition())
        closestInActiveGDistance=min(temp_distance,closestInActiveGDistance)
    else:
      closestInActiveGDistance = 0
    foodleft=len(foodlist)
    numberOfCapsulesLeft = len(currentGameState.getCapsules())
    nowscore=successorGameState.getScore()
    socre= -1.5 * closestDistancefood + \
          -2    * (1./closestActiveGDistance) + \
          -2    * closestInActiveGDistance + \
          -20.0 * numberOfCapsulesLeft + \
          -4    * foodleft+\
          1  * nowscore
    return socre

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    ans=self.value(gameState,0)
    return ans[0]

  def value(self,gameState,now_depth):
    if now_depth==self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose() :
        return (Directions.STOP, self.evaluationFunction(gameState))
    if now_depth%gameState.getNumAgents()==0:
        return self.maxfunc(gameState,now_depth)
    else:
       return self.minifunc(gameState,now_depth)
  def maxfunc(self,gameState,depth):
      actions=gameState.getLegalActions(0)
      if len(actions)==0:
        return (None,self.evaluationFunction(gameState))

      score=-float("inf")
      maxaction=(Directions.STOP,-float("inf"))
      for action in actions:
        nextState=gameState.generateSuccessor(0,action)
        nextAction=self.value(nextState,depth+1)
        if(nextAction[1]>score):
          score=nextAction[1]
          maxaction=(action,nextAction[1])
      return maxaction

  def minifunc(self,gameState,depth):
      actions=gameState.getLegalActions(depth%gameState.getNumAgents())
      if len(actions)==0:
        return (None,self.evaluationFunction(gameState))
      score=float("inf")
      miniAction=[None,score]
      for action in actions:
        nextState=gameState.generateSuccessor(depth%gameState.getNumAgents(),action)
        nextAction=self.value(nextState,depth+1)
        if(nextAction[1]<score):
          score=nextAction[1]
          miniAction=(action,nextAction[1])
      return miniAction




class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    alpha=-float("inf")
    beta=float("inf")
    ans=self.value(gameState,0,alpha,beta)
    return ans[0]

  def value(self,gameState,now_depth,alpha,beta):
    if now_depth==self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose() :
        return (None, self.evaluationFunction(gameState))
    if now_depth%gameState.getNumAgents()==0:
        return self.maxfunc(gameState,now_depth,alpha,beta)
    else:
       return self.minifunc(gameState,now_depth,alpha,beta)
  def maxfunc(self,gameState,depth,alpha,beta):
      actions=gameState.getLegalActions(0)
      if len(actions)==0:
        return (None,self.evaluationFunction(gameState))

      score=-float("inf")
      maxaction=(None,-float("inf"))
      for action in actions:
        nextState=gameState.generateSuccessor(0,action)
        nextAction=self.value(nextState,depth+1,alpha,beta)
        if(nextAction[1]>maxaction[1]):
          score=nextAction[1]
          maxaction=(action,nextAction[1])
        if(maxaction[1]>beta): 
          return maxaction
        alpha=max(alpha,maxaction[1])
      return maxaction

  def minifunc(self,gameState,depth,alpha,beta):
      actions=gameState.getLegalActions(depth%gameState.getNumAgents())
      if len(actions)==0:
        return (None,self.evaluationFunction(gameState))
      score=float("inf")
      miniAction=(None,score)
      for action in actions:
        nextState=gameState.generateSuccessor(depth%gameState.getNumAgents(),action)
        nextAction=self.value(nextState,depth+1,alpha,beta)
        if(nextAction[1]<score):
          score=nextAction[1]
          miniAction=(action,nextAction[1])
        if(miniAction[1]<alpha): 
          return miniAction
        beta=min(beta,miniAction[1])
        
      return miniAction
    # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    ans=self.value(gameState,0)
    return ans[0]

  def value(self,gameState,now_depth):
    if now_depth==self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose() :
        return (Directions.STOP, self.evaluationFunction(gameState))
    if now_depth%gameState.getNumAgents()==0:
        return self.maxfunc(gameState,now_depth)
    else:
       return self.expect(gameState,now_depth)
  def maxfunc(self,gameState,depth):
      actions=gameState.getLegalActions(0)
      if len(actions)==0:
        return (None,self.evaluationFunction(gameState))

      score=-float("inf")
      maxaction=(Directions.STOP,-float("inf"))
      for action in actions:
        nextState=gameState.generateSuccessor(0,action)
        nextAction=self.value(nextState,depth+1)
        if(nextAction[1]>score):
          score=nextAction[1]
          maxaction=(action,nextAction[1])
      return maxaction

  def expect(self,gameState,depth):
      actions=gameState.getLegalActions(depth%gameState.getNumAgents())
      if len(actions)==0:
        return (None,self.evaluationFunction(gameState))
      score=0.0
      for action in actions:
        nextState=gameState.generateSuccessor(depth%gameState.getNumAgents(),action)
        nextAction=self.value(nextState,depth+1)
        score=score+nextAction[1]
        # if(nextAction[1]<score):
        #   score=nextAction[1]
        #   miniAction=(action,nextAction[1])
      score=score/len(actions)
      return (None,score)
      # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: First of all, I set up the win and lose evaluation as positive and negarive infinite respectively.
    Then I find the cloest food and minus it with weight 2 to encourage the agent eat the food as more as possible. It can 
    also explain why I minus the left food with weight 6 to the final evaluation score. Thne I find the ghost within 3 step because
    the ghosts who are farther will not kill the pacman immediately. And then I added up all the distance of ghost within distance of 3 steps
    and minus it with weight 1.5 to avoid the immediate death of the pacman. The last consideraton is to get the count of left powerballs 
    and minus it with weight 3 so as to remind pacman if there are too many powerballs left to eat.
  """
  "*** YOUR CODE HERE ***"
  if currentGameState.isWin():
    return float("inf")
  if currentGameState.isLose():
    return -float("inf")
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  capsulelocations = currentGameState.getCapsules()

  score=scoreEvaluationFunction(currentGameState)

  closetfooddis=float("inf")
  foodPos=newFood.asList()
  for foodpos in foodPos:
    temp_dis=util.manhattanDistance(foodpos,newPos)
    closetfooddis=min(temp_dis,closetfooddis)

  cloestGhostdis=0
  count=0
  for ghost in newGhostStates:
    temp_dis = util.manhattanDistance(newPos, ghost.getPosition())
    if(temp_dis<3):
      cloestGhostdis=+temp_dis
  # cloestGhostdis=cloestGhostdis
  score-=cloestGhostdis*2
  score-=closetfooddis*1.5
  score-=6*len(foodPos)
  score-=3*len(capsulelocations)

  return score


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

