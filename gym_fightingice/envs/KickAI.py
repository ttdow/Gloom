from py4j.java_gateway import get_field

class KickAI(object):
  def __init__(self, gateway):
    self.gateway = gateway
    self.state = "STAND"

  def close(self):
    pass

  def getInformation(self, frameData, isControl, nonDelay):
    # Load the frame data every time getInformation gets called
    self.frameData = frameData
    self.cc.setFrameData(self.frameData, self.player)
		
  # please define this method when you use FightingICE version 3.20 or later
  def roundEnd(self, x, y, z):
    print(x)
    print(y)
    print(z)

  # please define this method when you use FightingICE version 4.00 or later
  def getScreenData(self, sd):
    pass
  
  def initialize(self, gameData, player):
    # Initializng the command center, the simulator and some other things
    self.inputKey = self.gateway.jvm.struct.Key()
    self.frameData = self.gateway.jvm.struct.FrameData()
    self.cc = self.gateway.jvm.aiinterface.CommandCenter()
    self.player = player
    self.gameData = gameData
    self.simulator = self.gameData.getSimulator()
    self.isGameJustStarted = True
    return 0

  def input(self):
    # The input is set up to the global variable inputKey
    # which is modified in the processing part
    return self.inputKey

  def getState(self):
    my = self.frameData.getCharacter(self.player)
    print(self)
    #print(type(my))

    #my_x = my.getX()
    #my_state = my.getState()
    #return my_state

  def processing(self):
    # First we check whether we are at the end of the round
    if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
      self.isGameJustStarted = True
      return
  
    if not self.isGameJustStarted:
      # Simulate the delay and look ahead 2 frames. The simulator class exists already in FightingICE
      self.frameData = self.simulator.simulate(self.frameData, self.player, None, None, 17)
      #You can pass actions to the simulator by writing as follows:
      #actions = self.gateway.jvm.java.util.ArrayDeque()
      #actions.add(self.gateway.jvm.enumerate.Action.STAND_A)
      #self.frameData = self.simulator.simulate(self.frameData, self.player, actions, actions, 17)
    else:
      # If the game just started, no point on simulating
      self.isGameJustStarted = False

    self.cc.setFrameData(self.frameData, self.player)
    
    distance = self.frameData.getDistanceX()
    
    my = self.frameData.getCharacter(self.player)
    my_x = my.getX()
    my_state = my.getState()
    self.state = my_state

    opp = self.frameData.getCharacter(not self.player)
    opp_x = opp.getX()
    opp_state = opp.getState()

    if self.cc.getSkillFlag():
      # If there is a previous "command" still in execution, then keep doing it
      self.inputKey = self.cc.getSkillKey()
      return
    
    # We empty the keys and cancel skill just in case
    self.inputKey.empty()
    self.cc.skillCancel()
    
    #self.cc.commandCall("")

  class Java:
    implements = ["aiinterface.AIInterface"]
