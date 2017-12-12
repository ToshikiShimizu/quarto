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

class PolicyGradientPlayer(ComputerPlayer):
    def __init__(self, name, size):
        super().__init__(name, size)
        self.batch = []
        self.history = []
        self.rewards = []
        self.this_result = None
        self.IN  = 1+2**(self.SIZE+1)+self.SIZE**2+self.SIZE**3
        self.OUT = 2**self.SIZE + self.SIZE**2
        self.mlp = MLP(self.IN, self.OUT)
        #self.optimizer = optimizers.RMSpropGraves(lr=0.0025)
        self.optimizer = optimizers.SGD(lr=0.0025)
        #self.optimizer = optimizers.Adam(0.0025)
        #self.optimizer = optimizers.MomentumSGD(lr=0.0025)
        self.optimizer.setup(self.mlp)
        self.gamma = 0.99
    def win(self):
        self.n_win += 1
        self.this_result = 1
    def lose(self):
        self.n_lose += 1
        self.this_result = -1
    def draw(self):
        self.n_draw += 1
        self.this_result = 0
    def decide_where_to_place(self, board, pieces):
        idx1 = np.random.choice(board.exist.shape[0])
        idx2 = np.random.choice(board.exist.shape[1])
        self.action_where = [idx1,idx2]

        self.decide_w(board,pieces,is_my_turn = True)
    def decide_what_to_place(self, board, pieces):
        idx = np.random.choice(len(pieces.exist))
        self.action_what = idx

        self.decide_w(board,pieces,is_my_turn = True)
    def decide_w(self,board,pieces,is_my_turn):
        state = pack_state(is_my_turn,pieces,board)
        self.get_state_code(state)
        x = np.array(list(self.state_code)).astype(np.float32).reshape(-1,self.IN)
        self.get_action_code(self.mlp.predict(x))
        self.get_action_w(is_my_turn,self.action_code)
        self.history.append([self.state_code,self.action_code])
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
        idx = np.random.choice(len(action.data[0]),1,p=action.data[0])
        temp = np.zeros(len(action.data[0]),dtype=np.int32)
        temp[idx] = 1
        self.action_code =  "".join(temp.astype(np.str))



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
    def end_process(self,board,pieces):
        self.add_batch()
        self.clear_history()
        if (self.n_win+self.n_lose+self.n_draw) % 1 == 0:
            self.update()
            self.clear_batch()

    def add_batch(self):
        reward = [self.this_result * (self.gamma ** (len(self.history) - i - 1)) for i in range(len(self.history))]
        self.batch.extend(self.history)
        self.rewards.extend(reward)

    def clear_batch(self):
        self.batch = []
        self.rewards = []
    def clear_history(self):
        self.history = []
    def update(self):
        self.batch = np.array(self.batch)

        self.mlp.cleargrads()
        self.x = np.array([list(code) for code in self.batch[:,0]]).astype(np.float32)
        self.target = np.array([list(code) for code in self.batch[:,1]]).astype(np.float32)

        self.target = np.argmax(self.target,axis=1).astype(np.int32)


        #ls = [[] for i in range(6)]
        # for i in range(len(self.x)):
        #
        #     loss = self.mlp(self.x[i].reshape(1,-1), self.target[i].reshape(1,-1))
        #     loss.backward()
        #     params = [self.mlp.l1.W.grad,self.mlp.l2.W.grad,self.mlp.l3.W.grad,self.mlp.l1.b.grad,self.mlp.l2.b.grad,self.mlp.l3.b.grad]
        #     for j in range(len(params)):
        #         ls[j].append(params[j] * self.rewards[i])
        # for j, param in enumerate(params):
        #     param = np.array(ls[j]).sum(axis=0)
        #
        # self.optimizer.update()

        loss = self.mlp(self.x, self.target)
        loss.backward()
        params = [self.mlp.l1.W.grad,self.mlp.l2.W.grad,self.mlp.l3.W.grad,self.mlp.l1.b.grad,self.mlp.l2.b.grad,self.mlp.l3.b.grad]

        for param in params:
            param *= self.this_result#episodeの行動すべてに対して等しい値を用いる(AlphaGo)

        self.optimizer.update()



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
        elif cap == "pg":
            players.append(PolicyGradientPlayer("pg"+n,size))

    return players[0],players[1]


def test_env(p1,p2,SIZE):
    test = Game(p1,p2,SIZE)
    for i in range(1000):
        test.play()
    print ("TEST_ENV")
    p1.show_result()
    p2.show_result()

if __name__=="__main__":
    f  = codecs.open('test.py', 'r', 'utf-8')
    source = f.read()
    np.random.seed(1)
    TRIAL = 1000000
    SIZE = 2
    p1,p2 = set_player("pg","pg",SIZE)
    SAVE = False
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

                p1.show_result()

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
