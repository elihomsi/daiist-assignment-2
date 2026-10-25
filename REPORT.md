# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**: Eli Homsi Valencia
- **Student ID**: 18938
- **Email**: ehomsi.ieu2022@student.ie.edu
- **Group**: BBADBA 

## Business problem, and the cost of being wrong

The model is intended to be used by insurance companies or for car rental companies, which photographs each car when a customer returns it. The model would cheack each photo for new body damage; then, if it detects a damage above certain confidence threshold, the car is flagged to have certain damage. Moreover, the image would be revised by a supervisor, and if the damage is real and new, the customer would be charged and sends the car for repair.

A false alarm for this model would represent a waste of time for the company's staff, that would have to revise a car that had no any new damage. On the other side, a missed damage would mean that a damaged car would be marked as clean, which could cost the company money since they can't bill any customer; there would not be any clear clue about which customer was the one who damaged, which will make this issue untracable and therefore, the damage would need to be covered by the company.

I assume a false alarm could costs about €3: roughly 5 minutes of a supervisor’s time at about €20 per hour (about €1.70), plus a small delay before the car can be rented out again. On the other side, a missed damage could costs, lets say, about €400, the average repair bill the company has to cover itself because it can no longer bill the customer who caused it.

In this sense, missed damage is therefore the worse error, because a review costs minutes of staff time while an unbilled repair costs hundreds of euros. Because of this, the confidence threshold should lean low, so the model flags more borderline cars and misses fewer real damages. A person must review every flagged image before any customer is charged. 


## Dataset

* Dataset : yolov8-z6snb, project damage-detect-pzgeu, V1. Link: https://universe.roboflow.com/yolov8-z6snb/damage-detect-pzgeu/dataset/1.

* Each image is a photo of one car, with a box around it labelled damaged or undamaged. In my pipeline, one image is one returned car, and the question is whether it shows damage. There are two classes: Damage-detected and Undamaged. My positive class is Damage-detected, because missing a damage is the costly mistake in my business problem. An image is flagged when the model finds at least one Damage-detected box above the confidence threshold.

* This dataset fits the problem because each photo shows one car with a label, so one photo is one decision, like one returned car, and it has an Undamaged class, so I can count false alarms as well as misses. Most car damage datasets only have damaged cars, so they cannot measure false alarms. The match is not perfect. The photos look like auction or stock images, not phone photos taken at a rental return. Also, many of the test cars are damaged, which is far more than a real rental business would see. So I treat the results as a best case and adjust them to realistic conditions in the cost analysis.


## What the dataset's composition implies

The dataset has 717 damaged and 895 undamaged cars in train, 138 and 272 in valid, and 92 and 119 in test. The three groups have 1,501, 366 and 206 photos. Both classes have enough examples, so no class is too small to trust. Each car box covers about 60% of the photo, so the cars are big and easy to see.

The test set is small. With 92 damaged cars, missing one car lowers the share of damaged cars caught by about 1%. So small differences between results may just be luck.

The test set also does not match real life. About half of its cars are damaged, but in a real rental business far fewer are. The photos are clean pictures of single cars, so I cannot say how the model works with phone photos, bad light or dirty cars.

## Finetuning and baselines

*Which model and task (detection or classification) and why. A results table on
the same test split: your finetuned model, the pretrained COCO model zero-shot,
and a trivial baseline. What changed, and whether the gap is as large as you
expected.*

## Per-class evaluation and failure modes

*Per-class AP50, or per-class precision/recall and the confusion matrix.
Which class carries the headline number, which one is dragging, and what the
model actually confuses with what. Show where it breaks, not just that it
works.*

## Cost matrix and operating point

*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

## What a person still has to check

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
