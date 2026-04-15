 Here is a draft of the dataset section:                                                                                                                                                      
                                                                                                                                                                                               
  ---             
  Dataset
                                                                                                                                                                                               
  Video Corpus
                                                                                                                                                                                               
  Our evaluation corpus comprises 2,688 video clips depicting interpersonal interactions, of which 2,239 contain aggressive behavior and 449 are non-aggressive controls (labeled "none").
  Videos were sourced from publicly available online platforms including YouTube, Facebook, Dailymotion, Instagram, and X (formerly Twitter), as well as from existing academic datasets
  including UCF-Crime and RWF-2000. The inclusion of videos from diverse sources ensures variation in recording quality, camera angle, resolution, and scene composition, reflecting the       
  heterogeneity of real-world surveillance and bystander footage.

  Annotation Schema                                                                                                                                                                            
   
  Each video was manually annotated with structured metadata describing the aggressive interaction. Annotations include: (1) the aggressive action being performed, drawn from a taxonomy of 21
   action categories; (2) a natural-language description of the aggressor; (3) a natural-language description of the victim; (4) the environment or setting; and (5) descriptions of any
  bystanders present but not directly participating. Person descriptions are appearance-based (e.g., "person in a dark jacket"), avoiding any identifying information as to reduce potential opportunities for bias.                         
                  
  The action taxonomy spans a range of physical and verbal aggression types. Percentages below are computed over the 2,239 aggressive videos. The most prevalent categories are punch (31.8%, n = 711), shove (14.8%, n = 331), kick (12.7%, n = 284), and hit with an object (9.3%, n = 209). Less frequent categories include hair grabbing, restraining, choking, slapping, dragging, and verbal aggression, among others. This distribution reflects the natural prevalence of these behaviors in the source material rather than an artificially balanced sampling.                                                                                 
                  
  Environment annotations are present for 52.7% of aggressive videos (n = 1,179) and span 296 unique location descriptions, with school-related settings (classrooms, hallways, locker rooms, cafeterias) and public spaces (streets, sidewalks, stores, restaurants) being the most common. Bystanders are annotated in 65.3% of aggressive videos (n = 1,462), with the majority described as "a group of people" when individual identification is not feasible. On average, each aggressive video contains 2.79 annotated individuals across all roles.                                
                  
  Benchmark Question Generation

  From the annotation corpus, we automatically generate a set of multiple-choice questions designed to probe distinct facets of video aggression understanding. Nine primary question types are
   organized into three tiers of increasing complexity:
                                                                                                                                                                                               
  Simple questions test individual attribute recognition: primary action identification (what aggressive action is occurring), aggressor identification (who is performing the aggression), and
   victim recognition (who is the target).
                                                                                                                                                                                               
  Compound questions require jointly identifying two attributes: aggressor and location, action and victim, and aggressor and victim. By combining attributes, these questions exponentially expand the answer space—correctly identifying one component but not the other still yields an incorrect response.
                                                                                                                                                                                               
  Complex questions demand holistic scene understanding: compound aggressor-action-victim (who did what to whom) and sequence verification (selecting the correct structured narrative of the  
  full interaction). An additional role identification question type presents a person description and asks the model to assign the correct role (aggressor, victim, or bystander).
                                                                                                                                                                                               
  For the full 2,688-video corpus, the benchmark yields 14,610 primary questions (approximately 5.4 per video on average). Most question types present 8 answer options (1 correct, 7 distractors), while role identification questions use 4 options given the constrained label space.
                                                                                                                                                                                               
  Distractor Design

  A central contribution of our benchmark is the design of distractors that resist shortcut reasoning. Several strategies are employed:                                                        
   
  1. Cross-video distractors. Distractor answers are drawn from real annotations of other videos in the dataset, ensuring that incorrect options are plausible and never obviously synthetic.  
  2. Same-video distractors. For person identification questions, other individuals visible in the same video (e.g., the victim when asking about the aggressor) are prioritized as
  distractors, since they cannot be eliminated by their absence from the footage.                                                                                                              
  3. Role reversals. In compound and sequence questions, the aggressor and victim are swapped to produce a distractor that tests whether the model understands directionality—who did what to 
  whom—rather than merely detecting the presence of specific individuals.                                                                                                                      
  4. Bystander-as-aggressor distractors. Bystanders are placed in the aggressor role to test whether models can distinguish participation from mere presence.
  5. Trick questions. With 10% probability, the correct answer is a negation option (e.g., "no aggressive action is taking place"), while all other choices are drawn from real cross-video    
  data. This prevents models from assuming aggression is always present and forces verification of visual content.                                                                             
  6. Semantic similarity filtering. Near-duplicate answers (e.g., multiple phrasings of "no one") are filtered to prevent models from exploiting repeated options as a signal.                 
  7. Answer position randomization. The position of the correct answer is shuffled uniformly across all questions to eliminate positional bias.                                                
                                                                                                                                                                                               
  The resulting correct-answer position distribution is approximately uniform (12.3%–12.9% across all eight positions for 8-option questions), confirming the absence of positional shortcuts.
                                                                                                                                                                                               
  ---                                                                                                                                                                                          
  This is a first draft. You'll likely want to adjust terminology to match your paper's conventions (e.g., whether you call it a "benchmark" vs. "dataset"), add citations for UCF-Crime and
  RWF-2000, and potentially add a table summarizing the action distribution or question type breakdown. Let me know if you'd like me to revise any section or add/remove detail.               
   
✻ Crunched for 2m 22s                                                                                                                                                                          
                  
❯ Excellent, now we are also working on a potential solution and this is what we came up with in a meeting. Give pros and cons to each as well as a recommendation to train an effective novel 
 model to detect the types of behavior we are looking at with this dataset: We are looking at two potential improvements which we could make. The first option would be adding Reinforcement   
Learning from AI or Human Feedback (RLAIF / RLHF) and the second would be adding Anchored Direct Preference Optimization (ADPO). Additionally, Chain of Thought (CoT) during the discussion.   
  ⎿  ⧉ Selected 1 lines from Untitled-1 in Visual Studio Code                                                                                                                                  
                                                                                                                                                                                             
●                                                  
