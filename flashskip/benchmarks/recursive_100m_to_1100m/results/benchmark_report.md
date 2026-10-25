# FlashSkip Recursive Checkpoint Benchmark: 100M–1100M

This is a deterministic local diagnostic suite using one shared tokenizer and the same recursive architecture at every checkpoint. It is intended for checkpoint comparison, not as a substitute for standardized external benchmarks. Core ranks use average ranks for exact ties.

**Best mean core rank:** 900M and 1100M (mean rank 3.12)
**Best handcrafted held-out perplexity:** 900M tokens (161.37)
**Best aggregate behavioral accuracy:** 900M, 1000M, 1100M (69.7%)

| Tokens | Logged val PPL | Held-out PPL | Grammar | Cloze | Facts | Common | Arith/Logic | Behavior avg | Repeat 3-gram | Core rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100M | 149.17 | 197.12 | 95% | 75% | 25% | 67% | 40% | 60.3% | 25.0% | 9.25 |
| 200M | 105.08 | 188.56 | 90% | 75% | 25% | 67% | 40% | 59.3% | 17.4% | 8.12 |
| 300M | 95.16 | 175.45 | 90% | 75% | 25% | 67% | 40% | 59.3% | 19.7% | 7.88 |
| 400M | 87.21 | 170.14 | 90% | 75% | 33% | 67% | 40% | 61.0% | 20.5% | 6.62 |
| 500M | 82.74 | 171.18 | 90% | 75% | 33% | 67% | 40% | 61.0% | 21.2% | 6.88 |
| 600M | 79.56 | 165.76 | 90% | 75% | 33% | 67% | 50% | 63.0% | 16.7% | 4.50 |
| 700M | 77.48 | 166.74 | 90% | 75% | 42% | 75% | 40% | 64.3% | 29.5% | 6.50 |
| 800M | 75.87 | 164.35 | 90% | 75% | 50% | 75% | 50% | 68.0% | 31.1% | 5.75 |
| 900M | 74.99 | 161.37 | 90% | 83% | 50% | 75% | 50% | 69.7% | 27.3% | 3.12 |
| 1000M | 74.40 | 162.75 | 90% | 83% | 50% | 75% | 50% | 69.7% | 28.8% | 4.25 |
| 1100M | 73.98 | 161.56 | 90% | 83% | 50% | 75% | 50% | 69.7% | 27.3% | 3.12 |

## Greedy generation samples

### 100M tokens
- **The capital of France is** →  the first to be the first to be the first to be the first to be the first to be the first to be
- **Once upon a time, a small robot** →  can be used to make a new device. The device is used to create a computer. The device is used to
- **To make a cup of tea,** →  and then to get a lot of food. The best way to eat is to eat a lot of food. 
- **Scientists discovered that the new material** →  was found in the field of the study. "The study was published in the journal Nature. The study was published
- **Alice went to the market because** →  the company was a big part of the world. The company was a company of the company, and the company was
- **The rain stopped, so the children** →  were not in the water. The water was the first to be a man. The water was the first to

### 200M tokens
- **The capital of France is** →  the first to be the first to be the first to be the first to be the first to be the first to be
- **Once upon a time, a small robot** →  can be used to make a new device. The first step is to use a computer to create a computer. 
- **To make a cup of tea,** →  and to make a cup of water, you can also be able to eat a lot of water, and you can also
- **Scientists discovered that the new material** →  was found in the atmosphere of the sun. The researchers found that the sun was a very high-quality, and
- **Alice went to the market because** →  of the fact that the number of people who have been in the market is not a matter of fact. The number
- **The rain stopped, so the children** →  were not going to go to the water. The water was not a good thing. The water was not a good

### 300M tokens
- **The capital of France is** →  the capital of the country. The capital of the country is the capital of the country. The capital is the
- **Once upon a time, a small robot** →  can be used to create a new device. The device is a device that can be used to create a computer.
- **To make a cup of tea,** →  you can use a cup of water to make a cup of water. - You can also use a cup of water
- **Scientists discovered that the new material** →  was used to produce a new new material. The researchers found that the material was used to produce a new material,
- **Alice went to the market because** →  of the fact that the world is not a matter of the world. The world is a world that is a world
- **The rain stopped, so the children** →  were not able to get the water. The water was not the same as the water. The water was not the

### 400M tokens
- **The capital of France is** →  the capital of the city of the city of the city of the city of the city of the city of the city of
- **Once upon a time, a small robot** →  can be used to create a new device. The first step is to use a computer to create a computer. The
- **To make a cup of tea,** →  you can use a cup of water to make a cup of water. - You can also use a cup of water
- **Scientists discovered that the new material** →  was used to produce a new new plant. The researchers found that the plant was a very important part of the plant
- **Alice went to the market because** →  of the fact that the world’s most popular and popular, and the most popular of all, the world’s
- **The rain stopped, so the children** →  were not in the water. The water was not the same as the water was in the water. The water was

### 500M tokens
- **The capital of France is** →  the capital of the city of the city of the city of the city of the city of the city of the city of
- **Once upon a time, a small robot** →  can be used to create a new device. The first step is to use a computer to create a computer. The
- **To make a cup of tea,** →  you can use a cup of tea to make a cup of tea. - You can also use a cup of tea
- **Scientists discovered that the new material** →  was found in the atmosphere. The researchers found that the water was not a source of water. The researchers found
- **Alice went to the market because** →  of the fact that the world is not a good thing. The world is a world that is not a world.
- **The rain stopped, so the children** →  were not going to go to the water. The water was not a good thing, but it was a good thing

### 600M tokens
- **The capital of France is** →  the capital of the country. The capital of the country is the capital of the country. The capital of the
- **Once upon a time, a small robot** →  can be used to create a new space. The first step is to create a new space. The first step is
- **To make a cup of tea,** →  you can use a cup of tea to make a cup of tea. - You can also use a cup of tea
- **Scientists discovered that the new material** →  was found in the atmosphere. "The first of the most important things is the discovery of the first human-like
- **Alice went to the market because** →  of the fact that the government was not the only way to make the country a new country. The government was not
- **The rain stopped, so the children** →  were not going to go to the water. The water was not the only way to keep the water in the water

### 700M tokens
- **The capital of France is** →  the capital of the city of the city of the city of the city of the city of the city of the city of
- **Once upon a time, a small robot** →  can be able to make a new space. The robot can be used to create a new space, and it can
- **To make a cup of tea,** →  you can use a cup of tea to make a cup of tea. - You can use a cup of tea to
- **Scientists discovered that the new material** →  was found in the atmosphere. The researchers found that the same material was found in the atmosphere, which was found in
- **Alice went to the market because** →  of the fact that the company was not a business. The company was a company that was a company that was a
- **The rain stopped, so the children** →  were not in the water. The water was not a good thing. The water was a good thing. 

### 800M tokens
- **The capital of France is** →  the capital of the city of the city of the city of the city of the city of the city of the city of
- **Once upon a time, a small robot** →  can be used to create a new machine. The robot can be used to create a new machine. The robot
- **To make a cup of tea,** →  you can use a cup of tea to make a cup of tea. - You can use a cup of tea to
- **Scientists discovered that the new material** →  was found in the atmosphere, and that the same kind of material is known to be the most important in the study.
- **Alice went to the market because** →  of the fact that the company was not a business. The company was not a business, but a company that was
- **The rain stopped, so the children** →  were not in the water. The water was not the only way to keep the water in the water. The

### 900M tokens
- **The capital of France is** →  the capital of the city of the city of the city of the city of the city of the city of the city of
- **Once upon a time, a small robot** →  can be used to create a new space. The robot can be used to create a new space, and it can
- **To make a cup of tea,** →  you can use a cup of tea to make a cup of tea. - You can use a cup of tea to
- **Scientists discovered that the new material** →  was found in the atmosphere. The researchers found that the same material was found in the atmosphere, which is the most
- **Alice went to the market because** →  of the fact that the company was not a business. The company was not a business, but a company that was
- **The rain stopped, so the children** →  were not going to go to the water. The water was not the only way to get the water from the water

### 1000M tokens
- **The capital of France is** →  the capital of the city of the city of the city of the city of the city of the city of the city of
- **Once upon a time, a small robot** →  can be used to create a new space. The robot can be used to create a new space, and it can
- **To make a cup of tea,** →  you can use a cup of tea to make a cup of tea. - You can use a cup of tea to
- **Scientists discovered that the new material** →  was found in the atmosphere. The researchers found that the same material was found in the atmosphere, which is the most
- **Alice went to the market because** →  of the fact that the company was not a business. The company was a business company, and the company was a
- **The rain stopped, so the children** →  were not going to go to the water. The water was not a good thing. The water was a good

### 1100M tokens
- **The capital of France is** →  the capital of the city of the city of the city of the city of the city of the city of the city of
- **Once upon a time, a small robot** →  can be used to create a new space. The robot can be used to create a new space, and it can
- **To make a cup of tea,** →  you can use a cup of tea to make a cup of tea. - You can use a cup of tea to
- **Scientists discovered that the new material** →  was found in the atmosphere. The researchers found that the same material was found in the atmosphere, which is the most
- **Alice went to the market because** →  of the fact that the company was not a business. The company was not a business, but a company that was
- **The rain stopped, so the children** →  were not going to go to the water. The water was not the only way to get the water from the water
