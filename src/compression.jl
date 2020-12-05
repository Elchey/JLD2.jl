const CODECZLIB_ID = UInt16(1)
const BLOSC_ID = UInt16(32001)

const SUPPORTED_COMPRESSIONLIBS = Dict(
    :codeczlib => UInt16(1),
    #:blosc => UInt16(32001),
    :lz4 => UInt16(32004),
    )

# For loading need filter_ids as keys

const REGISTERED_COMPRESSIONLIBS = Dict(
    UInt16(1) => (:CodecZlib, :ZlibCompressor, :ZlibDecompressor, :ZlibCompressorStream, :ZlibDecompressorStream),
    #32001 => (:Blosc, nothing, nothing)
    UInt16(32004) => (:CodecLz4, :LZ4FrameCompressor, :LZ4FrameDecompressor, :LZ4FrameCompressorStream, :LZ4FrameDecompressorStream)
)

issupported_filter(filter_id) = filter_id ∈ keys(REGISTERED_COMPRESSIONLIBS)

#############################################################################################################
# Dynamic Package Loading Logic copied from FileIO
const load_locker = Base.ReentrantLock()

is_installed(pkg::Symbol) = get(Pkg.installed(), string(pkg), nothing) != nothing

function _findmod(f::Symbol)
    for (u,v) in Base.loaded_modules
        (Symbol(v) == f) && return u
    end
    nothing
end
function topimport(modname)
    @eval Base.__toplevel__  import $modname
    u = _findmod(modname)
    @eval $modname = Base.loaded_modules[$u]
end

function checked_import(pkg::Symbol)
    lock(load_locker) do
        # kludge for test suite
        if isdefined(Main, pkg)
            m1 = getfield(Main, pkg)
            isa(m1, Module) && return m1
        end
        if isdefined(JLD2, pkg)
            m1 = getfield(JLD2, pkg)
            isa(m1, Module) && return m1
        end
        m = _findmod(pkg)
        isnothing(m) || return Base.loaded_modules[m]
        topimport(pkg)
        return Base.loaded_modules[_findmod(pkg)]
    end
end

#############################################################################################################

function get_compressor(f::JLDFile, ::Bool)
    # No specific compression lib was given. Return the default
    get_compressor(f, :codeczlib)
end


function get_compressor(::JLDFile{IOT}, c::Symbol) where {IOT<: Union{MmapIO, IOStream}}
    if c ∉ keys(SUPPORTED_COMPRESSIONLIBS)
        throw(ArgumentError("$c is not a supported compressor. Valid arguments are $(collect(keys(SUPPORTED_COMPRESSIONLIBS))...)"))
    end
    filter_id = SUPPORTED_COMPRESSIONLIBS[c]
    modname, compressorname, decompressorname, comprstreamname, decomprstreamname = REGISTERED_COMPRESSIONLIBS[filter_id]
    m = checked_import(modname)
    if IOT <: MmapIO
        compressor = @eval $m.$compressorname
        return filter_id, compressor
    else
        compressor = @eval $m.$comprstreamname
        return filter_id, compressor
    end
end

function get_decompressor(::JLDFile{IOT}, filter_id::UInt16) where {IOT<: Union{MmapIO, IOStream}}
    modname, compressorname, decompressorname, comprstreamname, decomprstreamname = REGISTERED_COMPRESSIONLIBS[filter_id]
    m = checked_import(modname)
    if IOT <: MmapIO
        @eval decompressor = $m.$decompressorname
        return decompressor
    else
        @eval decompressor = $m.$decomprstreamname
        return decompressor
    end
end

function deflate_pipeline_message(filter_id::UInt16)
    io = IOBuffer()
    write(io, HeaderMessage(HM_FILTER_PIPELINE, 12, 0))
    write(io, UInt8(2))                 # Version
    write(io, UInt8(1))                 # Number of Filters
    write(io, filter_id)                # Filter Identification Value (= deflate)
    write(io, UInt16(0))                # Flags
    write(io, UInt16(1))                # Number of Client Data Values
    write(io, UInt32(5))                # Client Data (Compression Level)
    take!(io)
end

const PIPELINE_MESSAGE_SIZE = length(deflate_pipeline_message(zero(UInt16)))

function deflate_data(f::JLDFile, data::Array{T}, odr::S, wsession::JLDWriteSession,
        compressor::Type{<:TranscodingStreams.Codec}) where {T,S}
    buf = Vector{UInt8}(undef, odr_sizeof(odr) * length(data))
    cp = Ptr{Cvoid}(pointer(buf))
    @simd for i = 1:length(data)
        @inbounds h5convert!(cp, odr, f, data[i], wsession)
        cp += odr_sizeof(odr)
    end
    transcode(compressor, buf)
end


@inline function read_compressed_array!(v::Array{T}, f::JLDFile{MmapIO},
                                        rr::ReadRepresentation{T,RR},
                                        data_length::Int,
                                        decompressor
                                        #::Type{<:TranscodingStreams.Codec}
                                        ) where {T,RR}
    io = f.io
    inptr = io.curptr
    data = transcode(decompressor, unsafe_wrap(Array, Ptr{UInt8}(inptr), data_length))
    @simd for i = 1:length(v)
        dataptr = Ptr{Cvoid}(pointer(data, odr_sizeof(RR)*(i-1)+1))
        if !jlconvert_canbeuninitialized(rr) || jlconvert_isinitialized(rr, dataptr)
            @inbounds v[i] = jlconvert(rr, f, dataptr, NULL_REFERENCE)
        end
    end
    io.curptr = inptr + data_length
    v
end

@inline function read_compressed_array!(v::Array{T}, f::JLDFile{IOStream},
                                        rr::ReadRepresentation{T,RR},
                                        data_length::Int,
                                        decompressorstream
                                        #::Type{<:TranscodingStreams.Codec}
                                        ) where {T,RR}
    io = f.io
    data_offset = position(io)
    n = length(v)
    data = read!(decompressorstream(io), Vector{UInt8}(undef, odr_sizeof(RR)*n))
    @simd for i = 1:n
        dataptr = Ptr{Cvoid}(pointer(data, odr_sizeof(RR)*(i-1)+1))
        if !jlconvert_canbeuninitialized(rr) || jlconvert_isinitialized(rr, dataptr)
            @inbounds v[i] = jlconvert(rr, f, dataptr, NULL_REFERENCE)
        end
    end
    seek(io, data_offset + data_length)
    v
end



@inline chunked_storage_message_size(ndims::Int) =
    sizeof(HeaderMessage) + 5 + (ndims+1)*sizeof(Length) + 1 + sizeof(Length) + 4 + sizeof(RelOffset)


function write_chunked_storage_message(
    io::IO,
    elsize::Int,
    dims::NTuple{N,Int},
    filtered_size::Int,
    offset::RelOffset) where N
    write(io, HeaderMessage(HM_DATA_LAYOUT, chunked_storage_message_size(N) - sizeof(HeaderMessage), 0))
    write(io, UInt8(4))                     # Version
    write(io, UInt8(LC_CHUNKED_STORAGE))    # Layout Class
    write(io, UInt8(2))                     # Flags (= SINGLE_INDEX_WITH_FILTER)
    write(io, UInt8(N+1))                   # Dimensionality
    write(io, UInt8(sizeof(Length)))        # Dimensionality Size
    for i = N:-1:1
        write(io, Length(dims[i]))          # Dimensions 1...N
    end
    write(io, Length(elsize))               # Element size (last dimension)
    write(io, UInt8(1))   #3                # Chunk Indexing Type (= Single Chunk)
    #write(io, UInt8(3))                     # Fixed Array Index
    #write(io, UInt8(8))                     # Number of bits needed for storing number of entries in data block page
    write(io, Length(filtered_size))        # Size of filtered chunk
    write(io, UInt32(0))                    # Filters for chunk
    write(io, offset)                       # Address
end


function write_compressed_data(cio, f, data, odr, wsession, filter_id, compressor)
    write(cio, deflate_pipeline_message(filter_id))
    # deflate first
    deflated = deflate_data(f, data, odr, wsession, compressor)

    write_chunked_storage_message(cio, odr_sizeof(odr), size(data), length(deflated), h5offset(f, f.end_of_data))
    write(f.io, end_checksum(cio))
   
    f.end_of_data += length(deflated)
    write(f.io, deflated)
end



