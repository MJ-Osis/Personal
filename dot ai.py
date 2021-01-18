import neat
import os
import random
import pickle

import pygame
pygame.init()
pygame.display.set_caption('DOT AI')

win_w = 500
win_h = 500

win = pygame.display.set_mode((win_w, win_h))

STAT_FONT = pygame.font.SysFont("comicsans", 50)

quit_game = False

GEN = 0

class Dot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.dir = 0

        self.size = 10

    def dir_is(self, dir):
        self.dir = dir

    def move(self, dir, vel):
        self.x += vel * dir

    def draw(self, win):
        pygame.draw.rect(win, (0, 0, 0), (self.x, self.y, self.size, self.size))


class Wall:

    def __init__(self, y, gap, vel):
        self.y = y
        self.height = 0
        self.gap = gap
        self.width = 40
        self.vel = vel

        self.passed = False
        self.set_height()

    def set_height(self):
        self.place = random.randrange(50, 350)

    def move(self):
        self.y += self.vel

    def draw(self, win):
        pygame.draw.rect(win, (0, 0, 0), (0, self.y, self.place, self.width))
        pygame.draw.rect(win, (0, 0, 0), (self.place + self.gap, self.y, win_w - self.place + self.gap, self.width))

    def collide(self, dot):

        if dot.x < self.place and dot.y + dot.size < self.y + self.width and dot.y > self.y:
            return True
        if dot.x + dot.size > self.place + self.gap and dot.y + dot.size < self.y + self.width and dot.y > self.y:
            return True
        else:
            return False


def drawG(win, dots, walls, score, gen, alive):
    pygame.draw.rect(win, (255, 255, 255), (0, 0, 500, 500))

    for wall in walls:
        wall.draw(win)

    text = STAT_FONT.render('Score: ' + str(score), 1, (0, 0, 0))
    win.blit(text, (win_w - 10 - text.get_width(), 10))

    text = STAT_FONT.render('Alive: ' + str(alive), 1, (0, 0, 0))
    win.blit(text, (10, 40))

    text = STAT_FONT.render('Gen: ' + str(gen), 1, (0, 0, 0))
    win.blit(text, (10, 10))

    for dot in dots:
        dot.draw(win)

    pygame.display.update()

def main(genomes, config):

    global GEN
    GEN += 1

    nets = []
    ge = []

    dots = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        dots.append(Dot(250, 400))
        g.fitness = 0
        ge.append(g)

    walls = [Wall(0, 100, 5)]
    win = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    score = 0

    run = True
    while run:

        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        wall_ind = 0
        if len(dots) > 0:
            if len(walls) > 1 and dots[0].y < walls[0].y:
                wall_ind = 1
        else:
            run = False
            break

        alive = len(dots)

        for x, dot in enumerate(dots):
            ge[x].fitness += 5

            in1 = dot.x
            in2 = abs(dot.x - walls[wall_ind].place)
            in3 = abs(dot.x - (walls[wall_ind].place + walls[wall_ind].gap))


            output = nets[x].activate((
                in1, in2, in3
                ))

            # if output[0] > 0.5:
            #     dot.move(1, 15)
            # elif output[0] < -0.5:
            #     dot.move(-1, 15)
            # else:
            #     ge[x].fitness += 5
            #     dot.move(0, 0)


            # if output[0] > 0.5:
            #     dot.move(5*output[0])
            # elif output[0] < -0.5:
            #     dot.move(5*output[0])
            # else:
            #     dot.move(0)

            # if output[1] > 0.5:
            #     ge[x].fitness -= 0.5
            #     dot.move(output[0]*5, 5)

            if output[0] > 0.5:
                ge[x].fitness -= abs(output[1])
                dot.move(1, abs(output[1] * 10))
            elif output[0] < -0.5:
                ge[x].fitness -= abs(output[1])
                dot.move(-1, abs(output[1] * 10))
            else:
                ge[x].fitness += 1
                dot.move(0, 0)

        # bird.move()
        add_wall = False
        rem = []
        for wall in walls:
            for x, dot in enumerate(dots):
                if wall.collide(dot):
                    ge[x].fitness -= 1
                    dots.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not wall.passed and wall.y > dot.y:
                    wall.passed = True
                    add_wall = True

            if wall.y + wall.width > win_w:
                rem.append(wall)

            wall.move()

        if add_wall:
            score += 1

            for g in ge:
                g.fitness -= 20

            walls.append(Wall(0, 100 - score, 5 + score * 3))

        for r in rem:
            walls.remove(r)

        for x, dot in enumerate(dots):
            if dot.x + dot.size >= 500 or dot.x <= 0:
                dots.pop(x)
                nets.pop(x)
                ge.pop(x)

        drawG(win, dots, walls, score, GEN, alive)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)
    # with open("winner.pkl", "wb") as f:
    #     pickle.dump(winner, f)
    #     f.close()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward_dot.txt')
    run(config_path)






