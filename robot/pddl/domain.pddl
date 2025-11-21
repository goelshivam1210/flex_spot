(define (domain robot-manipulation)
  
  (:requirements :strips :typing)
  
  (:types
    pickable-object interactive-object - object
    location
  )
  
  (:predicates
    (robot-at ?loc - location)
    (at ?obj - object ?loc - location)
    (holding ?obj - pickable-object)
    (hand-empty)
    (open ?obj - interactive-object)
    (closed ?obj - interactive-object)
    (inside ?obj - pickable-object ?container - interactive-object)
  )
  
(:action goto
  :parameters (?from ?to - location)
  :precondition (robot-at ?from)
  :effect (and (robot-at ?to) (not (robot-at ?from))))
  
  (:action pickup
    :parameters (?obj - pickable-object ?loc - location)
    :precondition (and (robot-at ?loc) (at ?obj ?loc) (hand-empty))
    :effect (and (holding ?obj) (not (hand-empty)) (not (at ?obj ?loc)))
  )
  
  (:action place
    :parameters (?obj - pickable-object ?loc - location)
    :precondition (and (robot-at ?loc) (holding ?obj))
    :effect (and (at ?obj ?loc) (hand-empty) (not (holding ?obj)))
  )
  
  (:action open
    :parameters (?obj - interactive-object ?loc - location)
    :precondition (and (robot-at ?loc) (at ?obj ?loc) (closed ?obj))
    :effect (and (open ?obj) (not (closed ?obj)))
  )
  
  (:action reveal
    :parameters (?obj - pickable-object ?container - interactive-object ?loc - location)
    :precondition (and (inside ?obj ?container) (open ?container) (at ?container ?loc))
    :effect (and (at ?obj ?loc) (not (inside ?obj ?container)))
  )
)