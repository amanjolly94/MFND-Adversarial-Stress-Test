import tkinter as tk

from tkinter import messagebox

import random

import itertools





class Minesweeper:

    def __init__(self, root, size=5, mines=5):

        self.root = root

        self.size = size

        self.mines = mines

        self.buttons = {}

        self.board = [[' ' for _ in range(size)] for _ in range(size)]

        self.mine_positions = set()

        self.game_over = False

        self.initialize_game()




    def initialize_game(self):

        # Place mines and initialize the board

        self.place_mines()

        self.create_buttons()




    def place_mines(self):

        while len(self.mine_positions) < self.mines:

            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)

            if (x, y) not in self.mine_positions:

                self.mine_positions.add((x, y))

                self.board[x][y] = '*'




    def create_buttons(self):

        for x in range(self.size):

            for y in range(self.size):

                button = tk.Button(

                    self.root, width=3, height=1, command=lambda x=x, y=y: self.on_click(x, y)

                )

                button.grid(row=x, column=y)

                self.buttons[(x, y)] = button




    def on_click(self, x, y):

        if self.game_over:

            return




        if (x, y) in self.mine_positions:

            self.reveal_mines()

            messagebox.showinfo("Game Over", "Boom! You hit a mine!")

            self.game_over = True

            return




        self.reveal_cell(x, y)




        if self.check_win():

            messagebox.showinfo("Congratulations", "You've cleared the board!")

            self.game_over = True




    def reveal_cell(self, x, y):

        if self.buttons[(x, y)]['state'] == 'disabled':

            return




        count = self.count_adjacent_mines(x, y)

        if count > 0:

            self.buttons[(x, y)].config(text=str(count), state='disabled', relief=tk.SUNKEN)

        else:

            self.buttons[(x, y)].config(state='disabled', relief=tk.SUNKEN)

            for dx, dy in itertools.product([-1, 0, 1], repeat=2):

                nx, ny = x + dx, y + dy

                if 0 <= nx < self.size and 0 <= ny < self.size:

                    self.reveal_cell(nx, ny)




    def count_adjacent_mines(self, x, y):

        count = 0

        for dx, dy in itertools.product([-1, 0, 1], repeat=2):

            if dx == 0 and dy == 0:

                continue

            nx, ny = x + dx, y + dy

            if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx][ny] == '*':

                count += 1

        return count




    def reveal_mines(self):

        for (x, y) in self.mine_positions:

            self.buttons[(x, y)].config(text='*', state='disabled', relief=tk.SUNKEN, bg='red')




    def check_win(self):

        for x in range(self.size):

            for y in range(self.size):

                if (x, y) not in self.mine_positions and self.buttons[(x, y)]['state'] != 'disabled':

                    return False

        return True





# Create the main window and start the game

if __name__ == "__main__":

    root = tk.Tk()

    root.title("Minesweeper")

    game = Minesweeper(root, size=8, mines=10)

    root.mainloop()