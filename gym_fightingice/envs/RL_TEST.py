
from py4j.java_gateway import get_field
import random
import torch

class RLTEST(object):
  def __init__(self, gateway):
    self.gateway = gateway
    #self.state = "STAND"
    self.state = ""
    self.action_to_perform = 0

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
    self.prev_state = None
    self.WakeUp = False
    self.frameCount = 0
    return 0

  def input(self):
    # The input is set up to the global variable inputKey
    # which is modified in the processing part
    return self.inputKey

  def getState(self):
    return self.state

  def setAction(self, action):
    self.action_to_perform = action

  def processing(self):

    _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
    actions_list = _actions.split(" ")
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
    #print(self.state)
    if self.state == self.prev_state:
        self.frameCount += 1
    else:
        self.frameCount = 0

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
    #print(self.action_to_perform, actions_list[self.action_to_perform])
    self.cc.commandCall(actions_list[self.action_to_perform])
    #if my_state.equals(self.gateway.jvm.enumerate.State.DOWN):
    #    self.cc.commandCall("STAND_F_D_DFB")
  class Java:
    implements = ["aiinterface.AIInterface"]