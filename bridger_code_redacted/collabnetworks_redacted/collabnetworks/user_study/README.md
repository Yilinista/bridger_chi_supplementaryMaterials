# Bridger User Study

A user study is conducted with a participant who is an author in the system. Each participant will have their relevant data prepared.

The base directory for a participant is `data_author{AuthorId}`, where `AuthorId` is the first MAG AuthorId (some people have multiple AuthorIds, many only have one).

This base directory will contain the files:
```
author{AuthorId}_card.json
author{AuthorId}_details.json
```
and the subdirectories:
```
simTask/
simMethod/
simTask_distMethod/
simMethod_distTask/
random/
simSpecter/
```

Each subdirectory will contain card and details json files for k other authors.