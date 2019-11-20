"""
This module contains agents that play reversi.

Version 3.0
"""

import boardgame2 as bg2
import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value
from math import inf
import numpy as np
import gym
np.inf


_ENV = gym.make('Reversi-v0')
_ENV.reset()
_MAX_DEPTH = 4

def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.

        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color

    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.

        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)
            p = Process(
                target=self.search,
                args=(
                    self._color, board, valid_actions,
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array(
                [output_move_row.value, output_move_column.value],
                dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.

        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains 
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for 
            `output_move_row.value` and `output_move_column.value` 
            as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            #time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


class BestAgent(ReversiAgent):

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        # time.sleep(3)

        move = self.minimax(board, _MAX_DEPTH, -np.inf, np.inf, self.player, color)
        output_move_row.value = move[0]
        output_move_column.value = move[1]

    def evaluate(self, state, color):
        # print('test',state)
        score = 0
        occur = 0



        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == color:
                    # simple case
                    score += 1
                    # corner case
                    if (0, 0) == (i, j) or (0, 7) == (i, j) or (i, j) or (7, 0) == (i, j) or (7, 7) == (i, j):
                        score += 10
                    elif (0, 2) == (i, j) or (0, 5) == (i, j) or (5, 0) == (i, j) or (5, 5) == (i, j):
                        score += 5
                elif state[i][j] == color*-1:
                    # simple case
                    score -= 1

                    # corner case
                    if (0, 0) == (i, j) or (0, 7) == (i, j) or (i, j) or (7, 0) == (i, j) or (7, 7) == (i, j):
                        score -= 10
                    elif (0, 2) == (i, j) or (0, 5) == (i, j) or (5, 0) == (i, j) or (5, 5) == (i, j):
                        score -= 5

        return score

    def minimax(self, state,  depth, alpha, beta, player, color):

        valids = _ENV.get_valid((state, self.player))
        valids = np.array(list(zip(*valids.nonzero())))
        if depth == 0:
            return [0, 0], -1, self.evaluate(state,color) - depth

        if player == self.player:
            best_score = -1, -np.inf
        else:
            best_score = -1, np.inf

        the_move = np.array([0, 0])
        for move in valids:
            board, turn = _ENV.get_next_state((state, player), move)

            val = self.minimax(board, depth - 1, alpha, beta, turn, color)

            if turn == self.player:

                if best_score[1] > val[1]:
                    best_score = val
                    the_move = move

                alpha = max(alpha, best_score[1])
                if alpha >= beta:
                    break
            else:
                if best_score[1] < val[1]:
                    best_score = val
                    the_move = move
                beta = min(beta, best_score[1])

                if alpha >= beta:
                    break

        if depth == _MAX_DEPTH:
            return the_move
        else:
            return best_score
