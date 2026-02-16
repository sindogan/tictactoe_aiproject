
import numpy as np
import random
import pickle
import os
from collections import defaultdict


class TicTacToe:
    """Tic-Tac-Toe game environment"""

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O

    def reset(self):
        """Reset the game board"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        """Get current board state as a tuple (for Q-table indexing)"""
        return tuple(self.board.flatten())

    def get_valid_moves(self):
        """Get list of valid moves (empty positions)"""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    moves.append((i, j))
        return moves

    def make_move(self, row, col):
        """Make a move and return (new_state, reward, done, winner)"""
        if self.board[row, col] != 0:
            return self.get_state(), -10, True, None  # Invalid move penalty

        self.board[row, col] = self.current_player
        winner = self.check_winner()
        done = winner is not None or len(self.get_valid_moves()) == 0

        # Calculate reward
        reward = 0
        if winner == 1:  # X wins
            reward = 10
        elif winner == -1:  # O wins
            reward = -10
        elif done:  # Draw
            reward = 1

        self.current_player *= -1  # Switch player
        return self.get_state(), reward, done, winner

    def check_winner(self):
        """Check if there's a winner"""
        # Check rows
        for row in self.board:
            if abs(sum(row)) == 3:
                return row[0]

        # Check columns
        for col in range(3):
            if abs(sum(self.board[:, col])) == 3:
                return self.board[0, col]

        # Check diagonals
        if abs(sum(self.board.diagonal())) == 3:
            return self.board[0, 0]
        if abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return self.board[0, 2]

        return None

    def print_board(self):
        """Print the current board state"""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("\n  0   1   2")
        for i in range(3):
            print(f"{i} {symbols[self.board[i, 0]]} | {symbols[self.board[i, 1]]} | {symbols[self.board[i, 2]]}")
            if i < 2:
                print("  --|---|--")
        print()


class QLearningAgent:
    """Q-Learning AI Agent for Tic-Tac-Toe"""

    def __init__(self, player_id, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.player_id = player_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))

    def get_action(self, state, valid_moves, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Exploration: random move
            return random.choice(valid_moves)
        else:
            # Exploitation: best known move
            best_value = float('-inf')
            best_moves = []

            for move in valid_moves:
                q_value = self.q_table[state][move]
                if q_value > best_value:
                    best_value = q_value
                    best_moves = [move]
                elif q_value == best_value:
                    best_moves.append(move)

            return random.choice(best_moves)

    def update_q_value(self, state, action, reward, next_state, next_valid_moves):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state][action]

        # Find maximum Q-value for next state
        max_next_q = 0
        if next_valid_moves:
            max_next_q = max([self.q_table[next_state][move] for move in next_valid_moves])

        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def save_model(self, filename):
        """Save Q-table to file"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filename):
        """Load Q-table from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    loaded_table = pickle.load(f)
                    self.q_table = defaultdict(lambda: defaultdict(float), loaded_table)
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class GameTrainer:
    """Training system for AI agents"""

    def __init__(self):
        self.agent_x = QLearningAgent(player_id=1, learning_rate=0.1, epsilon=0.3)
        self.agent_o = QLearningAgent(player_id=-1, learning_rate=0.1, epsilon=0.3)
        self.game = TicTacToe()

    def train(self, episodes=10000):
        """Train the AI agents"""
        print(f"Training AI agents for {episodes} episodes...")

        wins_x = 0
        wins_o = 0
        draws = 0

        for episode in range(episodes):
            self.game.reset()
            states = []
            actions = []
            rewards = []

            done = False
            while not done:
                current_state = self.game.get_state()
                valid_moves = self.game.get_valid_moves()

                if not valid_moves:
                    break

                # Current player chooses action
                if self.game.current_player == 1:
                    action = self.agent_x.get_action(current_state, valid_moves, training=True)
                else:
                    action = self.agent_o.get_action(current_state, valid_moves, training=True)

                # Store state and action
                states.append(current_state)
                actions.append(action)

                # Make move
                next_state, reward, done, winner = self.game.make_move(action[0], action[1])
                rewards.append(reward)

                # Update Q-values for the player who just moved
                if len(states) >= 2:
                    prev_state = states[-2]
                    prev_action = actions[-2]
                    prev_reward = rewards[-2]

                    if self.game.current_player == -1:  # X just moved
                        self.agent_x.update_q_value(prev_state, prev_action, prev_reward,
                                                    current_state, valid_moves)
                    else:  # O just moved
                        self.agent_o.update_q_value(prev_state, prev_action, prev_reward,
                                                    current_state, valid_moves)

                # Final update when game ends
                if done:
                    current_valid_moves = self.game.get_valid_moves()
                    if self.game.current_player == -1:  # X just moved
                        self.agent_x.update_q_value(current_state, action, reward,
                                                    next_state, current_valid_moves)
                    else:  # O just moved
                        self.agent_o.update_q_value(current_state, action, reward,
                                                    next_state, current_valid_moves)

                    # Count results
                    if winner == 1:
                        wins_x += 1
                    elif winner == -1:
                        wins_o += 1
                    else:
                        draws += 1

            # Decay epsilon for exploration and show progress
            if episode % 1000 == 0:
                self.agent_x.epsilon = max(0.01, self.agent_x.epsilon * 0.995)
                self.agent_o.epsilon = max(0.01, self.agent_o.epsilon * 0.995)
                if episode > 0:
                    print(f"Episode {episode}: X wins: {wins_x}, O wins: {wins_o}, Draws: {draws}")

        print(f"\nTraining completed!")
        print(f"Final results - X wins: {wins_x}, O wins: {wins_o}, Draws: {draws}")
        print(f"X win rate: {wins_x / episodes:.3f}")
        print(f"O win rate: {wins_o / episodes:.3f}")
        print(f"Draw rate: {draws / episodes:.3f}")

        # Save trained models
        if self.agent_x.save_model('agent_x.pkl') and self.agent_o.save_model('agent_o.pkl'):
            print("Models saved as 'agent_x.pkl' and 'agent_o.pkl'")
        else:
            print("Warning: Could not save models")


class GameTester:
    """Testing and gameplay system"""

    def __init__(self):
        self.agent_x = QLearningAgent(player_id=1, epsilon=0.0)  # No exploration during testing
        self.agent_o = QLearningAgent(player_id=-1, epsilon=0.0)
        self.game = TicTacToe()

    def load_agents(self):
        """Load trained agents"""
        x_loaded = self.agent_x.load_model('agent_x.pkl')
        o_loaded = self.agent_o.load_model('agent_o.pkl')

        if not x_loaded or not o_loaded:
            print("Warning: Could not load trained models. Please train first.")
            return False
        print("Trained models loaded successfully!")
        return True

    def test_ai_vs_ai(self, games=100):
        """Test AI vs AI performance"""
        if not self.load_agents():
            return

        print(f"\nTesting AI vs AI for {games} games...")

        wins_x = 0
        wins_o = 0
        draws = 0

        for game_num in range(games):
            self.game.reset()
            done = False

            while not done:
                valid_moves = self.game.get_valid_moves()
                if not valid_moves:
                    break

                current_state = self.game.get_state()

                if self.game.current_player == 1:
                    action = self.agent_x.get_action(current_state, valid_moves, training=False)
                else:
                    action = self.agent_o.get_action(current_state, valid_moves, training=False)

                _, _, done, winner = self.game.make_move(action[0], action[1])

                if done:
                    if winner == 1:
                        wins_x += 1
                    elif winner == -1:
                        wins_o += 1
                    else:
                        draws += 1

            # Show progress
            if (game_num + 1) % 20 == 0:
                print(f"Completed {game_num + 1}/{games} games...")

        print(f"\nTest results - X wins: {wins_x}, O wins: {wins_o}, Draws: {draws}")
        print(f"X win rate: {wins_x / games:.3f}")
        print(f"O win rate: {wins_o / games:.3f}")
        print(f"Draw rate: {draws / games:.3f}")


    def play_human_vs_ai(self):
        """Human vs AI gameplay"""
        if not self.load_agents():
            return

        print("\n=== Human vs AI Tic-Tac-Toe ===")
        print("You are X, AI is O")
        print("Enter your move as 'row col' (0-2 for both)")
        print("Example: '1 1' for center position")

        while True:
            self.game.reset()
            done = False

            while not done:
                self.game.print_board()
                valid_moves = self.game.get_valid_moves()

                if not valid_moves:
                    break

                if self.game.current_player == 1:  # Human turn
                    try:
                        move_input = input("Your move (row col): ").strip().split()
                        if len(move_input) != 2:
                            print("Invalid input! Enter two numbers separated by space.")
                            continue

                        row, col = int(move_input[0]), int(move_input[1])

                        if row < 0 or row > 2 or col < 0 or col > 2:
                            print("Invalid move! Row and column must be 0, 1, or 2.")
                            continue

                        if (row, col) not in valid_moves:
                            print("Invalid move! That position is already taken.")
                            continue

                        _, _, done, winner = self.game.make_move(row, col)

                    except (ValueError, IndexError):
                        print("Invalid input! Enter two numbers separated by space.")
                        continue

                else:  # AI turn
                    print("AI is thinking...")
                    current_state = self.game.get_state()
                    action = self.agent_o.get_action(current_state, valid_moves, training=False)
                    print(f"AI plays: {action[0]} {action[1]}")
                    _, _, done, winner = self.game.make_move(action[0], action[1])

            # Game over
            self.game.print_board()

            if winner == 1:
                print("🎉 You win!")
            elif winner == -1:
                print("🤖 AI wins!")
            else:
                print("🤝 It's a draw!")

            play_again = input("\nPlay again? (y/n): ").strip().lower()
            if play_again != 'y':
                break



def main():
    """Main function to run the Tic-Tac-Toe AI program"""
    print("=== Tic-Tac-Toe AI with Q-Learning ===")
    print("CENG 3511 - Artificial Intelligence Final Project")
    print("-" * 50)

    while True:
        print("\nChoose an option:")
        print("1. Train AI agents")
        print("2. Test AI vs AI")
        print("3. Play against AI")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            episodes = input("Enter number of training episodes (default 10000): ").strip()
            try:
                episodes = int(episodes) if episodes else 10000
            except ValueError:
                episodes = 10000
                print("Invalid input, using default 10000 episodes.")

            trainer = GameTrainer()
            trainer.train(episodes)

        elif choice == '2':
            games = input("Enter number of test games (default 100): ").strip()
            try:
                games = int(games) if games else 100
            except ValueError:
                games = 100
                print("Invalid input, using default 100 games.")

            tester = GameTester()
            tester.test_ai_vs_ai(games)

        elif choice == '3':
            tester = GameTester()
            tester.play_human_vs_ai()

        elif choice == '4':
            print("Goodbye!")
            break

        else:
            print("Invalid choice! Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()