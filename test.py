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
from mlp import CNN

import codecs
import seaborn as sns
import matplotlib.pyplot as plt
"""
PolicyGradientによるQuartoAI

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
            self.sub_player.illegal()
            self.sub_player.super_illegal()
            return False

        if pieces.exist[idx]==False:#illegal
            self.main_player.win()
            self.sub_player.lose()
            self.sub_player.illegal()

            return False

        pieces.selected[idx] = True
        pieces.exist[idx] = False
        return True

    def place(self,board,pieces,where):
        board.selected = where
        if board.selected is None:
            self.main_player.lose()
            self.main_player.illegal()
            self.main_player.super_illegal()
            self.sub_player.win()
            return False

        if board.exist[board.selected[0],board.selected[1]]:
            self.main_player.lose()
            self.main_player.illegal()
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
    def illegal(self):
        pass
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
        self.accumulated_params = None
        self.train_mode = True
        self.IN  = 1+2**(self.SIZE+1)+self.SIZE**2+self.SIZE**3
        if ONE_HOT_ATTRIBUTE:
            self.IN  = 1+2**(self.SIZE+1)+(self.SIZE**2)*(2**self.SIZE+1)#one-hot attribute
        self.OUT = 2**self.SIZE + self.SIZE**2
        if USE_CNN:
            self.IN = (self.SIZE**2)*(2**(self.SIZE+1)+2)
            self.n_channel = 4*self.IN * 2


            #self.mlp = CNN(self.n_channel, self.OUT, self.SIZE,chainer.initializers.HeNormal())
            self.mlp = CNN(self.IN, self.OUT, self.SIZE)

        else:
            self.mlp = MLP(self.IN, self.OUT)

        #self.optimizer = optimizers.RMSpropGraves(lr=0.0025)
        #self.optimizer = optimizers.SGD(lr=0.01)
        self.optimizer = optimizers.Adam(alpha=1e-4)#best
        #self.optimizer = optimizers.Adam(1e-5)
        #self.optimizer = optimizers.AdaGrad()
        #self.optimizer = optimizers.MomentumSGD(lr=1e-3)
        self.optimizer.setup(self.mlp)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

        if GPU>=0:
            chainer.cuda.get_device(GPU).use()
            self.mlp.to_gpu()

        self.gamma = 1#0.99
    def win(self):
        self.n_win += 1
        self.this_result = 1
    def lose(self):
        self.n_lose += 1
        self.this_result = -1
    def draw(self):
        self.n_draw += 1
        self.this_result = 0
    def illegal(self):
        pass
    def super_illegal(self):
        pass
        #self.this_result = -2#-2で上書き

    def decide_where_to_place(self, board, pieces):
        idx1 = np.random.choice(board.exist.shape[0])
        idx2 = np.random.choice(board.exist.shape[1])
        self.action_where = [idx1,idx2]

        self.decide_w(board,pieces,is_my_turn = True)
    def decide_what_to_place(self, board, pieces):
        idx = np.random.choice(len(pieces.exist))
        self.action_what = idx

        self.decide_w(board,pieces,is_my_turn = False)

    def get_legal_info(self):
        #print (self.state_code)
        self.legal_what = self.state_code[1:2**self.SIZE+1]
        self.legal_what = np.array(list(self.legal_what)).astype(np.int)

        self.illegal_where = self.state_code[1+2**(self.SIZE+1):1+2**(self.SIZE+1)+self.SIZE**2]
        self.illegal_where = np.array(list(self.illegal_where)).astype(np.int)
    def decide_w(self,board,pieces,is_my_turn):
        state = pack_state(is_my_turn,pieces,board)
        self.get_state_code(state)
        self.get_legal_info()
        state_code = self.state_code#CNNの場合書き換わってしまうのでバックアップ


        if ONE_HOT_ATTRIBUTE:
            self.modify_state_code()#one-hot attribute
            if USE_CNN:
                self.get_image()
        x = np.array(list(self.state_code)).astype(np.float32).reshape(-1,self.IN)

        x = xp.array(x)

        self.get_action_code(self.mlp.predict(x),state_code)

        # if self.state_code == "111100001000000000000":
        # if self.state_code == "111100000100011000000":#01000000を渡せば必勝
        #     print (self.state_code,self.action_code)
        # if self.state_code == "100100100110011000000":#正しく置けば必勝
        #     print (self.state_code,self.action_code)

        self.history.append([self.state_code,self.action_code])


        self.get_action_w(is_my_turn,self.action_code)

    def modify_state_code(self):
        a = self.state_code[:-(self.SIZE**2+self.SIZE**3)]
        b = self.state_code[-(self.SIZE**2+self.SIZE**3):]
        new_b = to_one_hot(b,self.SIZE)
        #print (len(self.state_code))
        self.state_code = a + new_b
        #print (len(self.state_code))

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

    def modify_action_prob(self,action_prob,state_code):
        """
        probの補正
        """
        epsilon = 1e-3#計算の安定化。合法手がすべて確率0で出力された場合ゼロ除算が発生する
        if state_code[0]=="1":#自分の番ならwhereのみ選べる
            action_prob += epsilon
            action_prob[:2**self.SIZE]=0
            action_prob[2**self.SIZE:]*=(self.illegal_where+1)%2
            # print (action_prob.dtype)
            # print (action_prob.sum())

            action_prob /= action_prob.sum()
            #print (action_prob)
        else:
            action_prob += epsilon
            action_prob[2**self.SIZE:]=0
            action_prob[:2**self.SIZE]*=self.legal_what
            action_prob /= action_prob.sum()
            #print (action_prob)

        return action_prob



    def get_action_code(self,action,state_code):
        action_prob = action.data[0]

        if GPU >= 0:
            action_prob = chainer.cuda.to_cpu(action_prob)
        if MODIFY_PROB:
            action_prob = self.modify_action_prob(action_prob,state_code)

        idx = np.random.choice(len(action.data[0]),1,p=action_prob)
        #print (idx)
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
        if self.train_mode:
            self.add_batch()
            self.clear_history()
            if (self.n_win+self.n_lose+self.n_draw) % Episode_size == 0:
                self.update()
                self.clear_batch()
        else:
            pass
        #print (self.n_win,self.n_draw,self.n_lose)

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
        if ONE_SAMPLE_PER_GAME:
            if len(self.batch)==0:#全てのゲームで初手に相手が反則した場合、一切の履歴がない
                pass
            else:
                idx = np.random.choice(len(self.batch))
                self.batch = self.batch[idx:idx+1]
                self.rewards = self.rewards[idx:idx+1]




        if len(self.batch)==0:#全てのゲームで初手に相手が反則した場合、一切の履歴がない
            pass
        else:
            self.batch = np.array(self.batch)
            self.mlp.cleargrads()
            self.x = np.array([list(code) for code in self.batch[:,0]]).astype(np.float32)
            self.target = np.array([list(code) for code in self.batch[:,1]]).astype(np.float32)
            self.target = np.where(self.target==1)[1].astype(np.int32)#networkに入力するために、one-hotからインデックスに変換
            self.x = xp.array(self.x)
            self.target = xp.array(self.target)
            loss = self.mlp(self.x, self.target)
            self.rewards = xp.array(self.rewards)
            loss = F.mean(loss*self.rewards)#rewardsを要素ごとにかける
            loss.backward()
            self.optimizer.update()
    def get_image(self):#CNN入力用に変換
        #print (self.state_code)
        #print (self.state_code[:1],self.state_code[1+2**self.SIZE:1+2**(self.SIZE+1)],self.state_code[-(2**self.SIZE+1)*(self.SIZE**2):])

        selected_attribute = np.array(list(self.state_code[1+2**self.SIZE:1+2**(self.SIZE+1)])).astype(np.int32)
        selected_attribute = np.insert(selected_attribute,0,(int(self.state_code[:1])+1)%2)#先頭にsub player flag
        #print (selected_attribute)
        temp = np.ones((self.SIZE,self.SIZE,1))
        selected_attribute = temp*selected_attribute.reshape(1,1,-1)
        #print (selected_attribute.shape)


        attribute = np.array(list(self.state_code[-(2**self.SIZE+1)*(self.SIZE**2):])).astype(np.int32)
        attribute  = attribute.reshape(self.SIZE,self.SIZE,-1)
        #print (attribute)
        #print (attribute.shape)
        image =  np.concatenate((selected_attribute, attribute), axis=2)
        #print (image.shape)
        #print (image)
        self.state_code = image
    def show(self):
        weight = self.mlp.l1.W.data
        if GPU >= 0:
            weight = chainer.cuda.to_cpu(weight)
        weight = weight.transpose(0,2,1,3)
        weight = weight.reshape(weight.shape[0]*weight.shape[1],-1)
        sns.heatmap(weight,cmap='gray')
        plt.show()



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

def to_one_hot(code,size):#existであれば文字列111を[0-10000000]に変換、そうでなければ[1-00000000]
    exist = code[:size**2]
    exist = np.array(list(exist))
    attribute = code[size**2:]
    one_hot = np.zeros((size**2,2**size+1),dtype=np.int)
    attribute = np.array(list(attribute)).reshape(-1,size).astype(np.int)
    w = [2**(size-i-1) for i in range(size)]
    for i,c in enumerate(attribute):
        if exist[i] == "0":
            one_hot[i][0] = 1
        else:
            one_hot[i][2**size-c.dot(w)] = 1
    return "".join(one_hot.flatten().astype(np.str))

def test_env(p1,p2,SIZE):
    test = Game(p1,p2,SIZE)
    for i in range(N_test):
        test.play()
    print ("TEST_ENV")
    p1.show_result()
    p2.show_result()

MODIFY_PROB = False
ONE_HOT_ATTRIBUTE = True
USE_CNN = True
ONE_SAMPLE_PER_GAME = False
Episode_size = 64#この数*各エピソードでの行動回数=バッチサイズ#128で勝率6割
N_test = 1000
test_freq = 10000
save_freq = 1000000
if __name__=="__main__":
    GPU = 0
    if GPU >= 0:
        import cupy as cp
        xp = cp
        cp.random.seed(0)
    else:
        xp = np
    f  = codecs.open('test.py', 'r', 'utf-8')
    source = f.read()
    np.random.seed(1)
    TRIAL = 10000000
    SIZE = 4
    p1,p2 = set_player("pg","l",SIZE)
    SAVE = True
    LOAD = False
    test_p1 = True
    test_p2 = False
    vs_Random = True
    vs_Legal = True
    if LOAD:
        p1 = joblib.load("4_64.pkl")
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
        for episode in range(TRIAL):
            #vs random or legal
            # game = Game(p1,p2,SIZE)
            # game.play()

            #pg vs pg
            if episode % 2 ==0:
                game = Game(p1,p2,SIZE)
            else:
                game = Game(p2,p1,SIZE)
            game.play()

            #self
            p2 = copy.deepcopy(p1)#本当は最初にコピーしたいが、そうするとgpu実行時にエラーがでてしまう
            if episode % save_freq == save_freq-1:
                p1.source = source
                joblib.dump(p1,"4_64_ch2t"+str(episode+1)+".pkl")


            if episode % test_freq == 0:
                print ("episode self _ch2t",SIZE,Episode_size,episode)

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
                        test_env(t1,t2,SIZE)
                if vs_Legal:
                    if test_p1:
                        t2 = LegalPlayer("l2",SIZE)
                        t1 = copy.deepcopy(p1)
                        t1.clear_result()
                        t1.train_mode = False
                        test_env(t1,t2,SIZE)
                    if test_p2:
                        t1 = LegalPlayer("l1",SIZE)
                        t2 = copy.deepcopy(p2)
                        t2.clear_result()
                        t2.train_mode = False
                        test_env(t1,t2,SIZE)
