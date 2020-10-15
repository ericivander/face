package main

import (
	"html/template"
	"image"
	"io"
	"io/ioutil"
	"log"
	"net/http"

	_ "image/jpeg"
	_ "image/png"

	pigo "github.com/esimov/pigo/core"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/vitali-fedulov/images"
)

type Template struct {
	templates *template.Template
}

func (t *Template) Render(w io.Writer, name string, data interface{}, c echo.Context) error {
	return t.templates.ExecuteTemplate(w, name, data)
}

func main() {
	// Echo Template
	t := &Template{
		templates: template.Must(template.ParseGlob("views/*.html")),
	}

	// Echo instance
	e := echo.New()
	e.Renderer = t

	// Middleware
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())

	// Routes
	e.GET("/", hello)
	e.GET("/index", index)
	e.POST("/detect", detectFace)
	e.POST("/hash", hashImage)
	e.POST("/compare", compareImages)

	// Start server
	e.Logger.Fatal(e.Start(":2234"))
}

func hello(c echo.Context) error {
	return c.String(http.StatusOK, "Hello World!")
}

func index(c echo.Context) error {
	return c.Render(http.StatusOK, "index", nil)
}

func detectFace(c echo.Context) error {
	// Source
	fileHeader, err := c.FormFile("image")
	if err != nil {
		return err
	}
	file, err := fileHeader.Open()
	if err != nil {
		return err
	}
	defer file.Close()

	cascadeFile, err := ioutil.ReadFile("cascade/facefinder")
	if err != nil {
		log.Fatalf("Error reading the cascade file: %v", err)
	}

	src, err := pigo.DecodeImage(file)
	if err != nil {
		log.Fatalf("Cannot open the image file: %v", err)
	}

	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	cParams := pigo.CascadeParams{
		MinSize:     20,
		MaxSize:     1000,
		ShiftFactor: 0.1,
		ScaleFactor: 1.1,

		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	pigo := pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := pigo.Unpack(cascadeFile)
	if err != nil {
		log.Fatalf("Error reading the cascade file: %s", err)
	}

	angle := 0.0 // cascade rotation angle. 0.0 is 0 radians and 1.0 is 2*pi radians

	// Run the classifier over the obtained leaf nodes and return the detection results.
	// The result contains quadruplets representing the row, column, scale and detection score.
	dets := classifier.RunCascade(cParams, angle)

	// Calculate the intersection over union (IoU) of two clusters.
	dets = classifier.ClusterDetections(dets, 0.2)

	return c.JSON(200, dets)
}

type imageHash struct {
	Hash []float32 `json:"hash"`
}

func hashImage(c echo.Context) error {
	// Source
	fileHeader, err := c.FormFile("image")
	if err != nil {
		return err
	}
	file, err := fileHeader.Open()
	if err != nil {
		return err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return err
	}

	hash, _ := images.Hash(img)

	return c.JSON(200, &imageHash{Hash: hash})
}

type imageCompare struct {
	Similar bool `json:"similar"`
}

func compareImages(c echo.Context) error {
	// Source
	fileHeader1, err := c.FormFile("image1")
	if err != nil {
		return err
	}
	file1, err := fileHeader1.Open()
	if err != nil {
		return err
	}
	defer file1.Close()

	img1, _, err := image.Decode(file1)
	if err != nil {
		return err
	}

	fileHeader2, err := c.FormFile("image2")
	if err != nil {
		return err
	}
	file2, err := fileHeader2.Open()
	if err != nil {
		return err
	}
	defer file2.Close()

	img2, _, err := image.Decode(file2)
	if err != nil {
		return err
	}

	hash1, point1 := images.Hash(img1)
	hash2, point2 := images.Hash(img2)

	similar := images.Similar(hash1, hash2, point1, point2)

	return c.JSON(200, &imageCompare{Similar: similar})
}
