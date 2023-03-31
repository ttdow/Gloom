import os
import platform
import random
import subprocess
import time
from multiprocessing import Pipe
import threading
from threading import Thread

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from py4j.java_gateway import (CallbackServerParameters, GatewayParameters, JavaGateway, get_field)

import gym_fightingice
from gym_fightingice.envs.gym_ai import GymAI
from gym_fightingice.envs.gym_ai_display import GymAIDisplay
from gym_fightingice.envs.Machete import Machete

def game_thread(env):

    print("game_thread() was called.")

    print("Inside try.")
    env.game_started = True
    print("env.game_starter = True")
    print("Env: " + str(type(env)))
    print("Env.manager: " + str(type(env.manager)))
    print(env.manager)
    print("Env.game_to_start: " + str(type(env.game_to_start)))
    print(env.game_to_start)
    env.manager.runGame(env.game_to_start)
    print("Success")

    #env.game_started = False
    #print("Please IGNORE the Exception above because of restart java game")

    print("game_thread() completed successfully.")


def start_up():
    print("stat_up() was called.")
    raise NotImplementedError("Come soon later")

# only used for development, so gym.make cannot make this
class FightingiceEnv_TwoPlayer(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, freq_restart_java=3, env_config=None, java_env_path=None, port=None, auto_start_up=False, frameskip=False, display=False, p2_server=None):
       
        # Get each action as a string in a list
        _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
        action_strs = _actions.split(" ")

        # gym.spaces.Space - superclass used to define observation and action spaces
        # gym.spaces.Box - a possibly unbounded box in R**n
        #   Specifically, a Box represents the Cartesian product of n closed intervals.
        #   Each interval has the form [low, hi].
        #   Shape specifieds the number of dimensions of the box.
        #   i.e. a 143-dimensional box where each dimension has length=1
        self.observation_space = spaces.Box(low=0, high=1, shape=(143,))

        # gym.spaces.Discrete - a space consisting of finitely many elements.
        #   i.e. { 0, ..., 55 }
        self.action_space = spaces.Discrete(len(action_strs))

        # Get OS name      
        system_name = "windows"

        # Check if FightingIce is installed correct
        if java_env_path == None:
            self.java_env_path = os.getcwd()
        else:
            self.java_env_path = java_env_path

        # Get path to "FightingICE.jar"
        start_jar_path = os.path.join(self.java_env_path, "FightingICE.jar")

        # Get path to "/data/"
        start_data_path = os.path.join(self.java_env_path, "data")

        # Get path to "/lib"
        start_lib_path = os.path.join(self.java_env_path, "lib")

        # Get path to "/lib/lwjgl/*" (wildcard?)
        lwjgl_path = os.path.join(start_lib_path, "lwjgl", "*")

        # Get path to "/lib/*" (wildcard?)
        lib_path = os.path.join(start_lib_path, "*")

        # Get path to "/lib/natives/windows/"
        start_system_lib_path = os.path.join(self.java_env_path, "lib", "natives", system_name)
        
        # Get path to "/lib/natives/windows/*" (wildcard?)
        natives_path = os.path.join(start_system_lib_path, "*")

        # Error check file paths, if all good pass
        if os.path.exists(start_jar_path) and os.path.exists(start_data_path) and os.path.exists(start_lib_path) and os.path.exists(start_system_lib_path):
            pass
        else:
            if auto_start_up is False:
                error_message = "FightingICE is not installed in {}".format(
                    self.java_env_path)
                raise FileExistsError(error_message)
            else:
                start_up()

        # Set port as argument or grab a random port
        if port:
            self.port = port
        else:
            try:
                import port_for
                self.port = port_for.select_random()  # select one random port for java env
            except:
                raise ImportError(
                    "Pass port=[your_port] when make env, or install port_for to set startup port automatically, maybe pip install port_for can help")

        # Get path to '/data/ai/'
        self.java_ai_path = os.path.join(self.java_env_path, "data", "ai")

        # Get path to '/data/ai/*'
        ai_path = os.path.join(self.java_ai_path, "*")

        # "FightingICE.jar;/lib/lwjgl/*;/lib/natives/windows/*;/lib/*;/data/ai/*"
        self.start_up_str = "{};{};{};{};{}".format(start_jar_path, lwjgl_path, natives_path, lib_path, ai_path)

        # TODO what?
        self.need_set_memory_when_start = True

        # Self explanatory
        self.game_started = False
        self.round_num = 0

        # TODO Frequency restart java?
        self.freq_restart_java = freq_restart_java

        # TODO FightingICE params
        self.frameskip = frameskip
        self.display = display

        # None for p1, p2_server pipe connection for p2
        self.p2_server = p2_server

    def _start_java_game(self):

        # Start the Java game
        print("Start java env in {} and port {}".format(self.java_env_path, self.port))

        # TODO Dunno what this is
        devnull = open(os.devnull, 'w')

        # This is true
        if self.need_set_memory_when_start:
            # -Xms1024m -Xmx1024m
            # Needs to be set in Windows
            self.java_env = subprocess.Popen(["java", "-Xms1024m", "-Xmx1024m", "-cp", self.start_up_str, "Main", "--port", str(self.port), "--py4j", "--fastmode",
                                          "--grey-bg", "--inverted-player", "1", "--mute", "--limithp", "400", "400"], stdout=devnull, stderr=devnull)
        else:
            self.java_env = subprocess.Popen(["java", "-cp", self.start_up_str, "Main", "--port", str(self.port), "--py4j", "--fastmode",
                                            "--grey-bg", "--inverted-player", "1", "--mute", "--limithp", "400", "400"], stdout=devnull, stderr=devnull)        # self.java_env = subprocess.Popen(["java", "-cp", "/home/myt/gym-fightingice/gym_fightingice/FightingICE.jar:/home/myt/gym-fightingice/gym_fightingice/lib/lwjgl/*:/home/myt/gym-fightingice/gym_fightingice/lib/natives/linux/*:/home/myt/gym-fightingice/gym_fightingice/lib/*", "Main", "--port", str(self.free_port), "--py4j", "--c1", "ZEN", "--c2", "ZEN","--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"])
        
        # Sleep 3s for Java starting, if your machine is slow, make it longer
        time.sleep(5)
    
    # --------------------------------- START GATEWAY -------------------------
    def _start_gateway(self, p1=GymAI, p2=GymAI):

        # The JavaGateway provides a bridge between Python and Java programs.
        #   When you create an instance of the JavaGateway class in a Python program
        #   it creates a Java gateway server that listens on a specified port for
        #   incoming requests from the Python program.
        #   The Java gateway server is responsible for communicating with the Java
        #   Virtual Machine (JVM) and forwarding requests from the Python program to
        #   the Java program.
        # The JavaGateway constructor takes two parameters:
        #   gateway_parameters = the configuration parameters for the Java gateway server
        #   callback_server_parameters = the configuration parameters for the callback server
        #       port=0 sets to random available port
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=self.port), 
                                   callback_server_parameters=CallbackServerParameters(port=0))
        
        # Get the port number for the above callback server (Java program) that was randomly assigned
        python_port = self.gateway.get_callback_server().get_listening_port()

        # Give the callback client (Python program) the port number for callbacks
        self.gateway.java_gateway_server.resetCallbackClient(
            self.gateway.java_gateway_server.getCallbackClient().getAddress(), python_port)
        
        # String representing a Java class which should have contain a main method that
        # serves as the entry point for the Java program.
        #   PyManager
        self.manager = self.gateway.entry_point

        # Check if the pipe is built - only P1 sets p1_server and calls this method
        if self.p1_server is None:
            raise Exception("Must call build_pipe_and_return_p2 and also make p2 env after gym.make() but before env.reset()")
        
        # Set the pipe to be the p1_server connection
        self.pipe = self.p1_server

        # TODO Currently set to not display but I'm fuzzy on the difference
        if self.display:
            self.p1 = GymAIDisplay(self.gateway, self.p1_client, self.frameskip)
            self.p2 = GymAIDisplay(self.gateway, self.p2_client, self.frameskip)
        else:
            # Instantiate p1-type (GymAI currently) and p2-type (GymAI currently)
            #   gateway = JavaGateway connection between this and Java program
            #   p1_client = p1 pipe for AI?
            #   p2_client = p1 pipe for other AI?
            #   frameskip = false
            self.p1 = p1(self.gateway, self.p1_client, self.frameskip)
            self.p2 = p2(self.gateway, self.p2_client, self.frameskip)

        # Register the AI with the Java program?
        self.manager.registerAI("P1", self.p1)
        self.manager.registerAI("P2", self.p2)

        self.game_to_start = self.manager.createGame("ZEN", "ZEN", "P1", "P2", self.freq_restart_java)

        self.game = Thread(target=game_thread, name="game_thread", args=(self, ))

        self.game.start()
        self.game_started = True
        self.round_num = 0

        print("_start_gateway() completed successfully.")

    # --------------------------------- BUILD PIPE ----------------------------
    # Must call this function after "gym.make()" but before "env.reset()"
    def build_pipe_and_return_p2(self):

        # Create pipe between gym_env_api and python_ai for java env
        if self.p2_server is not None:
            raise Exception("Can not build pipe again if env is used as p2 (p2_server set)")
        
        # Forms a 'pipe' between p1_server and p1_client (distinct processes) so they can communicate
        self.p1_server, self.p1_client = Pipe()

        # Forms a 'pipe' between p2_server and p2_client (distinct processes) so they can communicate
        self._p2_server, self.p2_client = Pipe() # p2_server should not be used in this env but another

        # TODO Return a connection to p2 pipe so the 2 envs can communicate?
        return self._p2_server # p2_server is returned to build a gym env for p2

    def _close_gateway(self):

        print("_close_gateway() was called.")

        self.gateway.close_callback_server()
        self.gateway.close()
        del self.gateway

    def _close_java_game(self):

        print("_close_java_game() was called.")

        self.java_env.kill()
        del self.java_env
        #self.pipe.close()
        #del self.pipe
        self.game_started = False

    # --------------------------------- RESET ---------------------------------
    def reset(self, p1=GymAI, p2=GymAI):

        print("reset() was called")

        # For P1
        if self.p2_server is None:

            # Start Java game if game is not started
            if self.game_started is False:
                try:
                    self._close_gateway()
                    self._close_java_game()
                except:
                    pass

                self._start_java_game()
                self._start_gateway(p1, p2)

                print(str(threading.current_thread().getName()) + " created the Java game.")

            # TODO Restart the Java game periodically for some reason?
            if self.round_num == self.freq_restart_java * 3:  # 3 is for round in one game
                try:
                    self._close_gateway()
                    self._close_java_game()
                    self._start_java_game()
                except:
                    raise SystemExit("Can not restart game")
                
                self._start_gateway()
        
        # For P2
        else:

            # Save pipe connection reference
            self.pipe = self.p2_server

            # TODO Sleep P2 thread for 10s if the game is just starting or periodically for some reason?
            if self.round_num == 0 or self.round_num == self.freq_restart_java * 3:
                time.sleep(10) # p2 wait 10s
                self.round_num = 0
                self.game_started = True

                # Now only P2 should call this
                #   Send "reset" to p2_client which p1 has a reference to?
                self.pipe.send("reset")

        print("Sent reset to pipe.")
        self.round_num += 1

        # P1
        obs = self.pipe.recv()

        print("Obs: " + str(type(obs)))
        print("reset() completed succesfully.")

        return obs

    def step(self, action):

        print("step() was called")

        # check if game is running, if not try restart
        # when restart, dict will contain crash info, agent should do something, it is a BUG in this version
        if self.game_started is False:
            dict = {}
            dict["pre_game_crashed"] = True
            return self.reset(), 0, None, dict

        self.pipe.send(["step", action])
        new_obs, reward, done, info = self.pipe.recv()
        return new_obs, reward, done, {}

    def render(self, mode='human'):
        # no need
        pass

    def close(self):
        if self.game_started and self.p2_server is None:
            self._close_java_game()

# ------------------------------------- UNUSED --------------------------------
def play_thread(env):

    print("play_thread() was called.")

    obs = env.reset()
    done = False
    while not done:
        new_obs, reward, done, _ = env.step(random.randint(0, 10))

if __name__ == "__main__":
    env1 = FightingiceEnv_TwoPlayer()
    p2_server = env1.build_pipe_and_return_p2()
    env2 = FightingiceEnv_TwoPlayer(p2_server=p2_server)

    while True:
        t1 = Thread(target=play_thread, name="play_thread1", args=(env1, ))
        t2 = Thread(target=play_thread, name="play_thread2", args=(env2, ))
        t1.start()
        t2.start()
        t1.join()
        t2.join()