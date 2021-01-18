import pickle

import neat
import os
import random
import math
import angles

import pygame
pygame.init()
pygame.display.set_caption('DOT AI')

win_w = 500
win_h = 500

win = pygame.display.set_mode((win_w, win_h))

STAT_FONT = pygame.font.SysFont("comicsans", 50)

z_img = pygame.image.load('pic/zombie_gun.png')
b_img = pygame.image.load('pic/bullet_gun.png')
base_img = pygame.image.load('pic/base_gun.png')


quit_game = False

GEN = 0

class Gun:
    def __init__(self, x, y, shoot_speed):
        self.x = x
        self.y = y
        self.speed = shoot_speed

        self.bullets = []

        self.img = base_img

        self.health = 50

    def draw(self, win, bullets):
        win.blit(self.img, (self.x, self.y))

        for b in bullets:
            b.draw(win)
            b.move()

        pygame.draw.rect(win, (255, 0, 0), (10, 70, 100, 20))
        pygame.draw.rect(win, (0, 255, 0), (10, 70, self.health * 2, 20))



    def shoot(self, angle):
        self.bullets.append(Bullet(self.x + 25, self.y + 25, angle, self.speed))

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


    def hit(self, enemy):
        base_mask = self.get_mask()
        enemy_mask = enemy.get_mask()

        offset = (round(self.x - enemy.x), round(self.y - enemy.y))

        c_point = enemy_mask.overlap(base_mask, offset)

        if c_point:
            self.health -= 10
            return True


class Bullet:
    def __init__(self, x, y, angle, vel):
        self.x = x
        self.y = y
        self.a = math.radians(angle * 360)
        self.vel = vel

        self.color = (0, 0, 0)
        self.radius = 5

        self.img = b_img

    def draw(self,win):
        win.blit(self.img, (self.x, self.y))

    def move(self):
        self.x += math.cos(self.a) * self.vel
        self.y += math.sin(self.a) * self.vel

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Zombie:
    def __init__(self, angle, speed, gun):
        self.a = math.radians(angle * 180)
        self.x = 250 + math.cos(self.a) * 300
        self.y = 250 + math.sin(self.a) * 300

        self.dif = math.sqrt(abs(self.x - gun.x) ** 2 + abs(self.y - gun.y) ** 2)

        self.vel = speed * (self.dif / 100)

        self.img = z_img

    def move(self):
        self.x -= math.cos(self.a) * self.vel
        self.y -= math.sin(self.a) * self.vel

    def draw(self, win):
        win.blit(self.img, (self.x, self.y))

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

    def collide(self, bullet):

        bullet_mask = bullet.get_mask()
        enemy_mask = pygame.mask.from_surface(self.img)

        offset = (round(self.x - bullet.x), round(self.y - bullet.y))

        c_point = bullet_mask.overlap(enemy_mask, offset)

        if c_point:
            return True
        return False

    def get_distance(self, base):
        x_dist = abs(base.x - self.x)
        y_dist = abs(base.y - self.y)

        return math.sqrt((x_dist ** 2) + (y_dist ** 2))

def spawn_enemy(enemys, base):

    e = Zombie(random.random(), 2, base)
    enemys.append(e)

def draw_game(win, gun, enemys, score, gen, creature):

    pygame.draw.rect(win, (192,192,192), (0, 0, win_w, win_h))

    gun.draw(win, gun.bullets)

    text = STAT_FONT.render('Score: ' + str(score), 1, (0, 0, 0))
    win.blit(text, (win_w - 10 - text.get_width(), 10))

    text = STAT_FONT.render('Gen: ' + str(gen), 1, (0, 0, 0))
    win.blit(text, (10, 10))

    text = STAT_FONT.render('Creature: ' + str(1 + creature), 1, (0, 0, 0))
    win.blit(text, (10, 40))

    for enemy in enemys:
        enemy.draw(win)

    pygame.display.update()


def main(genomes, config):

    global GEN
    GEN += 1

    nets = []
    ge = []

    guns = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        guns.append(Gun(225, 225, 10))
        g.fitness = 0
        ge.append(g)


    for x, gun in enumerate(guns):

        win = pygame.display.set_mode((win_w, win_h))

        enemys = []
        bullets = []
        clock = pygame.time.Clock()
        score = 0

        creature = x
        shoot_loop = 0

        spawn_timer = int((1 / (score + 1)) * 100)

        run = True
        while run:

            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()


            # Enemy Spawn ---------------------------------------------------------

            spawn_timer -= 1

            if spawn_timer == 0:
                spawn_timer = int((1 / (score + 1)) * 100)
                spawn_enemy(enemys, gun)

            # ----------------------------------------------------



            if gun.health <= 0:
                guns.pop(x)
                nets.pop(x)
                ge.pop(x)
                break

            ge[x].fitness += 1

            for z, enemy in enumerate(enemys):
                enemy.move()
                for y, bullet in enumerate(gun.bullets):
                    if enemy.collide(bullet):
                        ge[x].fitness += 100

                        enemys.pop(z)
                        gun.bullets.pop(y)

                        score += 1

                    if bullet.x > 500 or bullet.x < 0 or bullet.y > 500 or bullet.y < 0:
                        gun.bullets.pop(y)

                if gun.hit(enemy):
                    ge[x].fitness -= 10
                    if len(enemys) > 0:
                        enemys.pop(z)

            # Setting inputs ----------------------------------------

            if len(enemys) >= 3:

                in1_x = enemys[0].x
                in1_y = enemys[0].y
                in2_x = enemys[1].x
                in2_y = enemys[1].y
                in3_x = enemys[2].x
                in3_y = enemys[2].y

            elif len(enemys) == 2:

                in1_x = enemys[0].x
                in1_y = enemys[0].y
                in2_x = enemys[1].x
                in2_y = enemys[1].y
                in3_x = 0
                in3_y = 0

            elif len(enemys) == 1:

                in1_x = enemys[0].x
                in1_y = enemys[0].y
                in2_x = 0
                in2_y = 0
                in3_x = 0
                in3_y = 0

            else:

                in1_x = 0
                in1_y = 0
                in2_x = 0
                in2_y = 0
                in3_x = 0
                in3_y = 0

            #--------------------------------------------------

            # keys = pygame.key.get_pressed()
            #
            # if keys[pygame.K_SPACE]:
            #     gun.shoot(random.random())

            #The brain-----------------------------------------

            output = nets[x].activate((
                in1_x, in1_y, in2_x, in2_y, in3_x, in3_y
            ))

            if output[0] < 0.5:
                if shoot_loop <= 0:
                    gun.shoot(output[1])
                    shoot_loop = 5

            #---------------------------------------------------

            shoot_loop -= 1

            draw_game(win, gun, enemys, score, GEN, creature)



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

    winner = p.run(main, 20)
    with open("gun_winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward_gun.txt')
    run(config_path)


