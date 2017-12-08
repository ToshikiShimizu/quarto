import numpy as np
import copy
import itertools
import time
from sklearn.externals import joblib
import pandas as pd
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Variable
from chainer import optimizers
import argparse
from chainer import cuda
import os
from mlp import MLP
import codecs
"""
DQNの実装

https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/agent.py

"""
class Game:
    def __init__(self, p1, p2, SIZE):
        self.p1 = p1
        self.p2 = p2
        self.SIZE = SIZE
        self.MAX_TURN = min(2**SIZE,SIZE**2)


    def play(self):
        self.main_player = self.p1
        self.sub_player = self.p2
        board = Board(self.SIZE)
        pieces = Pieces(self.SIZE)
        self.main_player.ready_play()
        self.sub_player.ready_play()
        for self.turn in range(self.MAX_TURN):
            self.sub_player.decide_what_to_place(board, pieces)
            if self.give(board,pieces,self.sub_player.action_what) == False:
                break
            self.main_player.decide_where_to_place(board, pieces)
            if self.place(board,pieces,self.main_player.action_where) == False:
                break
            if self.is_finished(board):
                break
            self.switch_player()
        self.main_player.end_process(board,pieces)
        self.sub_player.end_process(board,pieces)




    def is_finished(self,board):
        if board.is_winning():
            self.main_player.win()
            self.sub_player.lose()
            return True
        elif self.MAX_TURN == self.turn + 1:
            self.main_player.draw()
            self.sub_player.draw()
            return True
        return False


    def give(self,board,pieces,what):
        idx = what
        if idx is None:#super illegal
            self.main_player.win()
            self.sub_player.lose()
            return False

        if pieces.exist[idx]==False:#illegal
            self.main_player.win()
            self.sub_player.lose()
            return False

        pieces.selected[idx] = True
        pieces.exist[idx] = False
        return True

    def place(self,board,pieces,where):
        board.selected = where
        if board.selected is None:
            self.main_player.lose()
            self.sub_player.win()
            return False

        if board.exist[board.selected[0],board.selected[1]]:
            self.main_player.lose()
            self.sub_player.win()
            return False
        board.exist[board.selected[0],board.selected[1]] = True
        idx = np.where(pieces.selected)[0]
        board.attribute[board.selected[0],board.selected[1]] = pieces.ATTRIBUTE[idx]
        pieces.selected  = np.zeros(2**self.SIZE,dtype=np.bool_)
        return True

    def switch_player(self):
        temp = self.sub_player
        self.sub_player = self.main_player
        self.main_player = temp

class Pieces:
    def __init__(self, SIZE):
        self.SIZE = SIZE
        self.selected = np.zeros(2**self.SIZE,dtype=np.bool_)
        self.selected_attribute = None
        self.exist = np.ones(2**self.SIZE,dtype=np.bool_)
        self.ATTRIBUTE = np.array(list(itertools.product([0,1], repeat=SIZE)),dtype=np.bool_)#属性のリスト Boolで[2**SIZE][SIZE]


class Board:
    def __init__(self, SIZE):
        self.SIZE = SIZE
        self.exist = np.zeros((self.SIZE,self.SIZE),dtype = np.bool_)
        self.attribute = np.zeros((SIZE,SIZE,SIZE),dtype = np.bool_)
        self.selected = None
    def is_winning(self):#勝利判定。揃っている列を見つけた時点でTrueを返す。
        def check_bingo(attribute,exist):#縦に見て、何かの属性が揃っていてかつ全て埋まっているか確認
            def check_all_equall(array):#何かの属性が揃っているか確認
                temp1 = np.logical_not(np.logical_or.reduce(array))
                temp2 = np.logical_and.reduce(array)
                temp = np.logical_or(temp1,temp2).T
                return (np.logical_or.reduce(temp))
            def check_all_exist(array):#全て埋まっているか確認
                temp = np.logical_and.reduce(array)
                return temp
            eq = check_all_equall(attribute)
            ex = check_all_exist(exist)
            temp = np.logical_and(eq,ex)
            return np.logical_or.reduce(temp)
        if check_bingo(self.attribute, self.exist):#縦に確認
            return True
        if check_bingo(self.attribute.transpose(1,0,2), self.exist.T):#横に確認
            return True
        temp1 = np.array(list(map(np.diag,self.attribute.transpose(2,0,1))))
        temp2 = np.array(list(map(np.diag,self.attribute[::-1].transpose(2,0,1))))
        attribute = np.array([temp1,temp2])
        exist = np.array([np.diag(self.exist),np.diag(self.exist[::-1])])
        if check_bingo(attribute.transpose(2,0,1), exist.T):#斜めに確認
            return True
        return False


    def print_attribute(self):
        bar = ["-" for i in range(self.SIZE)]
        bar = "".join(bar)
        temp = self.attribute.astype(np.int).astype(np.str)
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if self.exist[i][j]:
                    print ("".join(temp[i][j]),end=" ")
                else:
                    print (bar, end=" ")
            print ()

class Player:
    def __init__(self, name,size):
        self.name = name
        self.action_where = None #TODO おそらく不要
        self.n_win = 0
        self.n_lose = 0
        self.n_draw = 0
        self.SIZE = size
        self.epsilon = None
    def end_process(self,board,pieces):
        pass
    def decide_where_to_place(self, board, pieces):
        """
        decideはwhat,whereを返すのみで、pieces,boardのメンバ変数の書き換えを行わない
        """
        pass
    def decide_what_to_place(self, board, pieces):
        pass
    def win(self):
        self.n_win += 1
    def lose(self):
        self.n_lose += 1
    def draw(self):
        self.n_draw += 1
    def ready_play(self):
        pass
    def show_result(self):
        print (self.name,"WIN:",self.n_win)
        print (self.name,"DRAW:",self.n_draw)
        print (self.name,"LOSE:",self.n_lose)
        print ()
    def clear_result(self):
        self.n_win = 0
        self.n_lose = 0
        self.n_draw = 0

def get_idx(str_array,size):#二進str配列を十進数intに変換
    vec = [2 ** i for i in range(SIZE)]
    idx = str_array.astype(np.int).dot(vec[::-1])
    return idx

def partial_transform_board(data, magnitude):
    size = data.shape[0]
    argmax = magnitude.argmax()
    if argmax >= (size**2)/2:
        data = data[::-1]
    argmax = magnitude.argmax()
    if argmax % size >= 2:
        data = np.swapaxes(data,0,1)[::-1]

    argmax = magnitude.argmax()
    if size == 4:
        if argmax == 4 or argmax == 5:
            #print ("a")
            temp = data[[1,0,3,2]]
            temp = np.swapaxes(temp,0,1)
            temp = np.swapaxes(temp[[1,0,3,2]],0,1)
            data = temp
    return data

def transform_board(board_exist,board_attribute,action_where):
    """
    これで位置情報のパターン数を減らす
    """
    size = board_attribute.shape[0]
    temp = board_attribute.reshape(-1,size)
    magnitude = np.array([int("".join(i)) for i in temp.astype(np.int).astype(np.str)])

    board_exist = partial_transform_board(board_exist,magnitude)
    board_attribute = partial_transform_board(board_attribute,magnitude)
    action_where = partial_transform_board(action_where,magnitude)
    return board_exist,board_attribute,action_where

class HumanPlayer(Player):
    def __init__(self, name, size):
        super().__init__(name, size)
    def decide_where_to_place(self, board, pieces):
        board.print_attribute()
        idx = np.where(pieces.selected)[0]
        print  ("".join(pieces.ATTRIBUTE[idx][0].astype(np.int).astype(np.str)))
        while(1):
            idx1 = int(input("idx1:"))
            idx2 = int(input("idx2:"))
            if board.exist[idx1,idx2] == False:
                break
        self.action_where = [idx1,idx2]
    def decide_what_to_place(self, board, pieces):
        board.print_attribute()
        temp = ["".join(element) for element in pieces.ATTRIBUTE[np.where(pieces.exist)[0]].astype(np.int).astype(np.str)]
        print (temp)
        while(1):
            attribute = np.array(list(input("attribute:")))
            print (attribute)
            idx = get_idx(attribute,SIZE)
            print (idx)
            if pieces.exist[idx]:
                break
        self.action_what = idx


    def end_process(self,board,pieces):
        print ("#################")
        print ("Finish")
        print ("#################")
        board.print_attribute()
        print ("#################")



def pack_state(is_my_turn,pieces,board):
    return [\
    is_my_turn,\
    pieces.exist,\
    pieces.selected,\
    board.exist,\
    board.attribute\
    ]

class ComputerPlayer(Player):#コンピュータはデフォルトでは合法手の中からランダムに手を選ぶ
    def __init__(self, name, size):
        super().__init__(name, size)

class RandomPlayer(ComputerPlayer):
    def __init__(self, name, size):
        super().__init__(name, size)
    def decide_where_to_place(self, board, pieces):
        idx1 = np.random.choice(board.exist.shape[0])
        idx2 = np.random.choice(board.exist.shape[1])
        self.action_where = [idx1,idx2]
    def decide_what_to_place(self, board, pieces):
        idx = np.random.choice(len(pieces.exist))
        self.action_what = idx
class LegalPlayer(ComputerPlayer):
    def __init__(self, name, size):
        super().__init__(name, size)
    def decide_where_to_place(self, board, pieces):
        while(1):
            idx1 = np.random.choice(board.exist.shape[0])
            idx2 = np.random.choice(board.exist.shape[1])
            if board.exist[idx1,idx2] == False:
                break
        self.action_where = [idx1,idx2]
    def decide_what_to_place(self, board, pieces):
        while(1):
            idx = np.random.choice(len(pieces.exist))
            if pieces.exist[idx]:
                break
        self.action_what = idx
class DQNPlayer(ComputerPlayer):
    def __init__(self, name, size):
        super().__init__(name, size)
        self.IN  = 1+2**(self.SIZE+1)+self.SIZE**2+self.SIZE**3
        self.OUT = 2**self.SIZE + self.SIZE**2

        self.mlp = MLP(self.IN, self.OUT)
        #self.optimizer = optimizers.RMSpropGraves(lr=0.0025)
        self.optimizer = optimizers.SGD(lr=0.0025)
        #self.optimizer = optimizers.Adam(0.0025)
        #self.optimizer = optimizers.MomentumSGD(lr=0.0025)
        self.optimizer.setup(self.mlp)
        self.target_mlp = copy.deepcopy(self.mlp)
        self.time = 0
        self.REPLAY_MEMORY_SIZE = 1000*1000
        self.TARGET_FREQUENCY = 10000#10000
        self.BATCH_SIZE = 32
        self.train_mode = True
        str_size = max(self.IN,self.OUT)
        dim = len(["s","a","r","s_dash"])
        self.D = np.zeros((self.REPLAY_MEMORY_SIZE,dim),dtype='|U'+str(str_size))
        self.epsilon = 1
        self.START_SIZE = 50000#50000 >=BATCH_SIZE

        self.STATE = 0
        self.ACTION = 1
        self.REWARD = 2
        self.S_DASH = 3
        self.cnt_legal = 0
        self.cnt_illegal = 0

        self.gamma = 0.99
        self.r_j = 0
    def win(self):
        self.n_win += 1
        self.reward = 1
    def lose(self):
        self.n_lose += 1
        self.reward = -1
    def draw(self):
        self.n_draw += 1
        self.reward = 0

    def store_transition(self,transition):#TODO change name
        if self.train_mode:
            self.D[self.time%self.REPLAY_MEMORY_SIZE] = transition

            self.time += 1
            if self.time == self.START_SIZE:
                self.START_EPISODE = episode
            if self.time > self.START_SIZE:
                self.epsilon = 1-((episode - self.START_EPISODE)*0.9)/(TRIAL - self.START_EPISODE)#TODO episode,TRIALの呼び出しを後で直す
            if self.time % self.TARGET_FREQUENCY == 0:
                self.target_mlp = copy.deepcopy(self.mlp)
        else:
            self.epsilon = 0


    def update(self):
        #print ("update")
        current_memory_size = min(self.time,self.REPLAY_MEMORY_SIZE)
        #print ("c:",current_memory_size)
        if current_memory_size >= self.START_SIZE:
            """
            start_sizeに達するまではrandom-policy,以降はepsilon-greedy
            """
            idx = np.random.choice(current_memory_size,self.BATCH_SIZE,replace=False).astype(np.int)
            self.get_target(self.D[idx])
            self.get_one_hot(self.D[idx])
            self.mlp.cleargrads()
            loss = self.mlp(self.x, self.target,self.one_hot)
            loss.backward()
            self.optimizer.update()
    def get_target(self,data):

        #もしs_dashがENDであればrの値をそのままtarget_vectorとして上書きする

        self.x = np.array([list(code) for code in data[:,self.STATE]]).astype(np.float32)
        t = self.target_mlp.predict(self.x).data
        max_t = np.max(t,axis=1)
        # if (np.count_nonzero(t - max_t.reshape(-1,1)) != self.BATCH_SIZE * (2**self.SIZE + self.SIZE ** 2 -1)):
        #     print ("multi max")#あくまで最大値がわかればよいので問題はない
        mask = (data[:,self.S_DASH]=="END")
        self.target = ((self.gamma *max_t + self.r_j)* np.logical_not(mask)  +  data[:,self.REWARD].astype(np.int) * mask).astype(np.float32).reshape(-1,1)

    def get_one_hot(self,data):
        self.one_hot = np.array([list(code) for code in data[:,self.ACTION]]).astype(np.float32)


    def end_process(self,board,pieces):
        if not(self.state_code_old is None):
            if self.train_mode:
                self.store_transition([self.state_code_old,self.action_code_old,str(self.reward),"END"])
                self.update()
            else:
                pass
    def decide_where_to_place(self, board, pieces):
        self.decide_w(board,pieces,is_my_turn = True)


    def decide_what_to_place(self, board, pieces):
        self.decide_w(board,pieces,is_my_turn = False)

    def decide_w(self,board,pieces,is_my_turn):
        state = pack_state(is_my_turn,pieces,board)

        self.get_state_code(state)
        x = np.array(list(self.state_code)).astype(np.float32).reshape(-1,self.IN)
        self.get_action_code(self.mlp.predict(x))


        # if self.state_code == "011110000000000000000":
        #     print (self.action_code)
        #     print (self.mlp.predict(x))

        if np.random.rand() < self.epsilon:
            temp = np.zeros(len(self.action_code),dtype=np.int).astype(np.str)
            temp[np.random.randint(0,len(self.action_code))]="1"
            self.action_code = "".join(temp)


        if not(self.state_code_old is None):
            self.store_transition([self.state_code_old,self.action_code_old,str(self.reward),self.state_code])
            self.update()
        self.state_code_old = self.state_code
        self.action_code_old = self.action_code




        self.get_action_w(is_my_turn,self.action_code)


    def get_action_w(self,is_my_turn,action_code):
        #what,whereの順に記録されている
        if is_my_turn:
            temp = action_code[2**self.SIZE:].find("1")

            if temp == -1:
                self.action_where = None
            else:
                self.action_where = [int(temp/self.SIZE),temp%self.SIZE]
        else:
            self.action_what = action_code.find("1")
            if self.action_what >= 2**self.SIZE:
                self.action_what = None#Noneで上書き
            else:
                pass#そのまま



    def get_state_code(self,state):#stateからkeyを生成
        IS_MY_TURN = 0
        PIECES_EXIST = 1
        PIECES_SELECTED = 2
        BOARD_EXIST = 3
        BOARD_ATTRIBUTE = 4

        size = state[BOARD_ATTRIBUTE].shape[0]
        f = ["" for i in range(len(state))]

        f[IS_MY_TURN] = str(int(state[IS_MY_TURN]))
        f[PIECES_EXIST] = "".join(state[PIECES_EXIST].astype(np.int).astype(np.str))
        f[PIECES_SELECTED] = "".join(state[PIECES_SELECTED].astype(np.int).astype(np.str))
        f[BOARD_EXIST] = "".join(state[BOARD_EXIST].astype(np.int).astype(np.str).flatten())
        f[BOARD_ATTRIBUTE] = "".join(state[BOARD_ATTRIBUTE].astype(np.int).astype(np.str).flatten())
        self.state_code = "".join(f)

    def get_action_code(self,action):
        idx = argmax(action.data[0])#predictorは2次元配列を返すので、最初のベクトルを指定する必要がある
        temp = np.zeros(len(action.data[0]),dtype=np.int32)
        temp[idx] = 1
        self.action_code =  "".join(temp.astype(np.str))


    def ready_play(self):
        self.state_code_old = None
        self.action_code_old = None
        self.reward = 0


def argmax(ls):
    max_value = np.max(ls)#単にargmaxとした場合max_valueが複数あった際にidxが小さいものが選ばれてしまう。特に初期は価値が0ばかりなので問題がある
    idx_max = np.where(ls == max_value)[0]#そこでmax_valueを持つidxの中からrandom_choiceをする
    selected_idx = np.random.choice(idx_max)
    return selected_idx

def set_player(cap1,cap2,size):
    ls_cap = [cap1,cap2]
    players = []
    for i,cap in enumerate(ls_cap):
        n = str(i+1)
        if cap == "h":
            players.append(HumanPlayer("h"+n,size))
        elif cap == "r":
            players.append(RandomPlayer("r"+n,size))
        elif cap == "l":
            players.append(LegalPlayer("l"+n,size))
        elif cap == "d":
            players.append(DQNPlayer("d"+n,size))

    return players[0],players[1]


def test_env(p1,p2,SIZE):
    test = Game(p1,p2,SIZE)
    for i in range(1000):
        test.play()
    print ("TEST_ENV")
    p1.show_result()
    p2.show_result()

if __name__=="__main__":
    f  = codecs.open('dqn.py', 'r', 'utf-8')
    source = f.read()
    np.random.seed(1)
    TRIAL = 1000000#1000000
    SIZE = 4
    p1,p2 = set_player("d","d",SIZE)
    SAVE = True
    LOAD = False
    test_p1 = True
    test_p2 = False
    vs_Random = True
    vs_Legal = True
    if LOAD:
        p1 = joblib.load("p1.pkl")
        #p2 = joblib.load("p2.pkl")
        p1.clear_result()
        p1.train_mode = False
        p2.clear_result()
        p2.train_mode = False
        game = Game(p1,p2,SIZE)
        game.play()
        p1.show_result()
        p2.show_result()
    else:
        game = Game(p1,p2,SIZE)
        for episode in range(TRIAL):
            game.play()
            if episode % 2000 == 0:
                print (p1.epsilon)
                p1.show_result()
                print (p2.epsilon)
                p2.show_result()
                if vs_Random:
                    if test_p1:
                        t2 = RandomPlayer("r2",SIZE)
                        t1 = copy.deepcopy(p1)
                        t1.clear_result()
                        t1.train_mode = False
                        test_env(t1,t2,SIZE)
                    if test_p2:
                        t1 = RandomPlayer("r1",SIZE)
                        t2 = copy.deepcopy(p2)
                        t2.clear_result()
                        t2.train_mode = False
                        test_env
                if vs_Legal:
                    if test_p1:
                        t2 = LegalPlayer("l2",SIZE)
                        t1 = copy.deepcopy(p1)
                        t1.clear_result()
                        t1.train_mode = False
                        test_env(t1,t2,SIZE)
                    if test_p2:
                        t1 = RandomPlayer("r1",SIZE)
                        t2 = copy.deepcopy(p2)
                        t2.clear_result()
                        t2.train_mode = False
                        test_env(t1,t2,SIZE)
        if SAVE:
            p1.source = source
            p2.source = source
            p1.D = None
            p2.D = None
            joblib.dump(p1,"p1"+".pkl")
            joblib.dump(p2,"p2"+".pkl")
