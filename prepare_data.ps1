$ErrorActionPreference = "Stop"

$TargetDir = Join-Path -Path $PSScriptRoot -ChildPath "trainer/data"

if (Test-Path -Path $TargetDir) {
    Remove-Item -Path "$TargetDir\*" -Recurse -Force
} else {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

$BaseUrl = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
$Files = @(
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
)

foreach ($file in $Files) {
    $gzFile = "$file.gz"
    $url = "$BaseUrl/$gzFile"
    $gzPath = Join-Path -Path $TargetDir -ChildPath $gzFile
    $outPath = Join-Path -Path $TargetDir -ChildPath $file

    curl.exe -sSL $url -o $gzPath

    $fileBytes = [System.IO.File]::ReadAllBytes($gzPath)
    if ($fileBytes.Length -lt 2 -or $fileBytes[0] -ne 0x1F -or $fileBytes[1] -ne 0x8B) {
        throw
    }

    $inFileStream = [System.IO.File]::OpenRead($gzPath)
    $outFileStream = [System.IO.File]::Create($outPath)
    $gzipStream = New-Object System.IO.Compression.GZipStream($inFileStream, [System.IO.Compression.CompressionMode]::Decompress)

    $gzipStream.CopyTo($outFileStream)

    $gzipStream.Close()
    $outFileStream.Close()
    $inFileStream.Close()

    Remove-Item -Path $gzPath -Force
}