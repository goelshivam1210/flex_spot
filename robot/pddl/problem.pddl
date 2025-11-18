(define (problem door-drawer-item)
  (:domain robot-manipulation)
  
  (:objects
    door drawer - interactive-object
    item - pickable-object
    start-loc door-loc drawer-loc - location
  )
  
  (:init
    (robot-at start-loc)
    (hand-empty)
    (at door door-loc)
    (at drawer drawer-loc)
    (inside item drawer) 
    (closed door)
    (closed drawer)
  )
  
  (:goal
    (holding item)
  )
)