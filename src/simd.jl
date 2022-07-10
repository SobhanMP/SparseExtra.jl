Base.argmax(vx::Vec{N}) where N = 
let m = vmaximum(vx) == vx,
    u = getfield(m, :u)
    trailing_zeros(u) + 1
end